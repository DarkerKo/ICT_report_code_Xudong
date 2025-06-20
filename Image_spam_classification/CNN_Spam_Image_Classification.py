from CNNModel import imageModel

import torch
from torch import nn
from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch.nn.functional as F
class Trainer:
    def __init__(self, ROOT_TRAIN, ROOT_TEST, GOAL_PATH):
        self.ROOT_TRAIN = ROOT_TRAIN
        self.ROOT_TEST = ROOT_TEST
        self.GOAL_PATH = GOAL_PATH
        self.pic_class = [cls for cls in os.listdir(ROOT_TRAIN)]
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            self.normalize])


        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize])
        self.train_dataset = ImageFolder(self.ROOT_TRAIN, transform=self.train_transform)
        self.test_dataset = ImageFolder(self.ROOT_TEST, transform=self.test_transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=48, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=48, shuffle=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

        self.model = imageModel(num=2).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.loss_train = []
        self.acc_train = []

        self.loss_test = []
        self.acc_test = []
    def train(self):
        loss, current, n = 0.0, 0.0, 0

        for batch, (x, y) in enumerate(self.train_dataloader):
            image, y = x.to(self.device), y.to(self.device)
            output = self.model(image)
            a = torch.sigmoid(output)
            b = F.one_hot(y, num_classes=2).to(torch.float32)
            cur_loss = F.binary_cross_entropy(a, b)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

        train_loss = loss / n
        train_acc = current / n
        print('train_loss: ' + str(train_loss))
        print('train_accuracy: ' + str(train_acc))

        return train_loss, train_acc
    def test(self):
        self.model.eval()
        loss, current, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.test_dataloader):
                image, y = x.to(self.device), y.to(self.device)
                output = self.model(image)

                a = torch.sigmoid(output)
                b = F.one_hot(y, num_classes=2).to(torch.float32)
                cur_loss = F.binary_cross_entropy(a, b)
                _, pred = torch.max(output, axis=1)
                cur_acc = torch.sum(y == pred) / output.shape[0]
                loss += cur_loss.item()
                current += cur_acc.item()
                n += 1

        test_loss = loss / n
        test_acc = current / n
        print('train_loss： ' + str(test_loss))
        print('train_accuracy： ' + str(test_acc))
        self.model.train()
        return test_loss, test_acc
    def matplot_loss(self, train_loss, val_loss):
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss, label='test_loss')
        plt.legend(loc='best')
        plt.ylabel('loss')
        plt.xlabel("epoch")
        plt.title("Comparison of loss values ​​of training set and validation set")

    def matplot_acc(self, train_acc, val_acc, max_acc, epoch):
        plt.plot(train_acc, label='train_acc')
        plt.plot(val_acc, label='test_acc')
        plt.legend(loc='best')
        plt.ylabel('acc/loss')
        plt.xlabel('epoch')
        plt.title('Comparison chart of accuracy and loss')
        plt.savefig(os.path.join(self.GOAL_PATH, f'plot_{max_acc}_{epoch}.png'))

    def main_train(self, epoch):
        min_acc = 0
        old_acc = 0
        for t in range(epoch):
            print(f"Epoch {t + 1}, Learning Rate: {self.lr_scheduler.get_last_lr()}")
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            self.lr_scheduler.step()

            print(f"epoch_{t+1}， train_loss:{train_loss}， train_acc:{train_acc}")
            self.loss_train.append(train_loss)
            self.acc_train.append(train_acc)
            self.loss_test.append(test_loss)
            self.acc_test.append(test_acc)
            if test_acc > min_acc:
                min_acc = test_acc
                print(f'save best model, epoch{t+1}')
                if os.path.exists(os.path.join(self.GOAL_PATH, f'best_model_{epoch}_{format(old_acc, "0.3f")}_.pkl')):
                    os.remove(os.path.join(self.GOAL_PATH, f'best_model_{epoch}_{format(old_acc, "0.3f")}_.pkl'))

                old_acc = test_acc
                torch.save(self.model.state_dict(), os.path.join(self.GOAL_PATH, f'best_model_{epoch}_{format(test_acc, "0.3f")}_.pkl'))
        self.matplot_loss(self.loss_train, self.loss_test)
        self.matplot_acc(self.acc_train, self.acc_test, max_acc=min_acc, epoch=epoch)
        plt.clf()
        print("Done！！！")
        return self
    def reset(self):
        self.loss_train.clear()
        self.loss_test.clear()
        self.acc_train.clear()
        self.acc_test.clear()
        return self.pic_class

if __name__ == '__main__':
    ROOT_TRAIN = "./DataSet_small/train"
    ROOT_TEST = "./DataSet_small/test"
    GOAL_PATH = "./myModel"

    trainer = Trainer(ROOT_TRAIN, ROOT_TEST, GOAL_PATH)
    epoch = 20
    trainer.main_train(epoch)



