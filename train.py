from train_and_eva import Train_and_test

epoches = 1
batch_size = 2
learning_rate = 1e-5

train = Train_and_test(epoches, batch_size, learning_rate, pretrained=True)
train.fit()