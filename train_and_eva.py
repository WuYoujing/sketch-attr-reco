import torch.nn.functional as F
import torch
import torch.optim as optim
import json
import copy
import time
import pandas as pd
import numpy as np

from model import Sum_model
from utils import FS2KDATA, set_traansform, get_loader

label_train_path = 'FS2K/anno_train.json'
label_test_path = 'FS2K/anno_test.json'
data_root_path = 'FS2K/sketch'


class Train_and_test():
    def __init__(self, epoches, batch_size, learning_rate, pretrained=True):
        self.epoches = epoches
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.label_train_path = label_train_path
        self.label_test_path = label_test_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained = pretrained
        self.transform = set_traansform()
        self.train_loader = get_loader(self.label_train_path, self.batch_size, 'train',
                                       self.transform)
        self.test_loader = get_loader(self.label_test_path, self.batch_size, 'test',
                                      self.transform)
        self.model = Sum_model(pretrained).to(self.device)
        self.optimer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, epoch):
        self.model.train()
        loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            img, label = data
            img = img.to(self.device)
            hair = self.model(img)

            hair_loss = F.cross_entropy(input=hair, target=label.to(self.device))
            hair_loss.backward()
            self.optimer.step()

            self.optimer.zero_grad()
            loss += hair_loss.item()

            if (batch_idx + 1) % (len(self.train_loader) // 4) == 0:
                print("Epoch: %d/%d, training batch_idx:%d ,  loss: %.4f" % (
                    epoch, self.epoches, batch_idx + 1, hair_loss.item()))
        # 返回epoch_loss
        return loss / (batch_idx + 1)

    def evaluate(self):
        self.model.eval()

        correct_dict = 0
        predict_dict = []
        label_dict = []

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                img, labels = data
                img = img.to(self.device)
                hair = self.model(img)

                for i in range(len(hair)):
                    pred = np.argmax(hair[i].data.numpy())
                    true_label = labels.data.numpy()[i]
                    if pred == true_label:
                        correct_dict += 1
                    predict_dict.append(pred)
                    label_dict.append(true_label)
        correct_dict = correct_dict * 100 / (len(self.test_loader) * self.batch_size)

        return correct_dict, predict_dict, label_dict

    def fit(self, model_path=None):
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print("模型参数文件加载：{}".format(model_path))

        best_model = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        train_loss = []
        eval_acc = []

        for epoch in range(self.epoches):
            running_loss = self.train(epoch)
            print("Epoch: %d, loss: %.4f, lr:%.7f" % (epoch, running_loss, self.learning_rate))
            correct, predict, label = self.evaluate()
            print("Epoch:{} accuracy:{}".format(epoch, correct))
            train_loss.append(running_loss)
            eval_acc.append(correct)

            if correct > best_acc:
                best_acc = correct
                best_model = copy.deepcopy(self.model.state_dict())
                best_predict = predict
                best_label = label

        acc_csv = pd.DataFrame(eval_acc, index=[i for i in range(self.epoches)])
        acc_csv.to_csv("./train_result/evalaccper_epoch.csv")
        model_path = "./train_result/model_param.pth"
        torch.save(best_model, model_path)

        report_dict = {}
        report_dict["best_mAP"] = best_acc
        report_dict["lr"] = self.learning_rate
        report_dict["optim"] = 'Adam'
        report_dict['Batch_size'] = self.batch_size
        report_json = json.dumps(report_dict)
        report_file = open("./train_result/report.json", 'w')
        report_file.write(report_json)
        report_file.close()
        print("完成")
