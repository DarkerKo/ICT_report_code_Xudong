import torch
from torch import nn
import torch.nn.functional as F


class imageModel(nn.Module):
    def __init__(self, num=2):
        super(imageModel, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)  # 3: RGB
        self.elu = nn.ELU()  # ELU함수는 입력이 음수인 경우에도 gradient가 0이 되는 문제를 완화할 수 있다.

        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)

        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()  # 1차원 tensor로 바꾼다.

        self.f6 = nn.Linear(4608, 2048)
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, num)

    def forward(self, x):  # x.shape = (batch_size, 3, Height, Width) = (1， 3， 224， 224）
        x = self.elu(self.c1(x))  # （1, 48， 55， 55）
        x = self.elu(self.c2(x))  # (1, 128, 55, 55)
        x = self.s2(x)  # (1, 128, 27, 27)
        x = self.elu(self.c3(x))  # （1， 192， 27， 27）
        x = self.s3(x)  # （1， 192， 13， 13）
        x = self.elu(self.c4(x))  # （1， 192， 13， 13）
        x = self.elu(self.c5(x))  # [1, 128, 13, 13]
        x = self.s5(x)  # [1, 128, 6, 6]

        x = self.flatten(x)  # [1,4608]

        x = self.f6(x)  # [1, 2048]
        x = F.dropout(x, p=0.5, training=True)
        x = self.f7(x)  # [1, 2048]
        x = F.dropout(x, p=0.5, training=True)
        x = self.f8(x)  # [1, 1000]
        x = F.dropout(x, p=0.5, training=True)

        x = self.f9(x)  # [1, 2]
        return x


if __name__ == '__main__':
    x = torch.rand([1, 3, 224, 224])
    model = imageModel()

    y = model(x)
    print(y)  # tensor([[0.0141, 0.0314]], grad_fn=<AddmmBackward0>)

