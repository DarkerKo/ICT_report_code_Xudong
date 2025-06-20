
import torch
import torch.nn as nn
import os

from torchtext.data.functional import to_map_style_dataset
import matplotlib.pyplot as plt
import sys
import logging
logging.basicConfig(level = logging.WARN,
                    stream = sys.stdout,
                    format = "%(asctime)s (%(module)s:%(lineno)d %(levelname)s: %(message)s",)



VOCAB_SIZE = 31650
class Basic_LSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, hidden_dim=128, lstm_layer=3, output_dim=2):
        super(Basic_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.lstm = nn.LSTM(embedding_dim , hidden_dim, lstm_layer, bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, batch_data):
        x = self.embedding(batch_data)

        out, _ = self.lstm(x)

        output = self.fc(out[:, -1, :])
        return output
class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, hidden_dim=128, lstm_layer=3, output_dim=2):
        super(CNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(32, hidden_dim, lstm_layer, bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)


    def forward(self, batch_data):
        x = self.embedding(batch_data)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        output = self.fc(out[:, -1, :]) # # torch.Size([32, 2])
        return output
def getStopWords():
    file = open('./stopwords_English.txt', 'r', encoding='utf-8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words

from torchtext.data import get_tokenizer

def dataParse(text, stop_words):
    tokenizer = get_tokenizer("basic_english")
    result  = tokenizer(text)
    words = [i for i in result if not i in stop_words]
    return words

def getData(inputFile):
    import csv
    stop_words = getStopWords()
    contents = []
    labels = []
    with open(inputFile, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            _, content, label = row
            content = dataParse(content, stop_words)
            if len(content) <= 0:
                continue

            contents.append(content)
            labels.append(int(label))

    return labels, contents
data_path = './DataSet_shuffle.csv'
total_labels, total_contents = getData(data_path)


ham = 0
spam = 0
for i in total_labels:
    if i == 0:
        ham += 1
    if i == 1:
        spam += 1
print("ham:", ham, "spam:", spam)

print('Data loading, cleaning, and word segmentation are completed!')
print(f"The number of tags is:{len(total_labels)}, The number of emails is:{len(total_contents)}")

from sklearn.model_selection import train_test_split
X_train, X_test_temple, train_y, test_temple_y = train_test_split(total_contents, total_labels, test_size=0.3, random_state=42)
X_val, X_test, val_y, test_y = train_test_split(X_test_temple, test_temple_y, test_size=0.5, random_state=42)

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    global total_labels, total_contents, X_train, train_y, X_val, val_y
    def __init__(self, type_dataSet):
        if type_dataSet == "total_data":
            self.myData = zip(total_labels, total_contents)
        elif type_dataSet == 'train':
            self.myData = zip(train_y, X_train)
        else:
            self.myData = zip(val_y, X_val)

    def __iter__(self):
        return iter(self.myData)
total_data_iter = CustomDataset("total_data")
tokenizer = get_tokenizer("basic_english")


def yield_tokens(train_data_iter):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        yield comment

from torchtext.vocab import build_vocab_from_iterator
vocab = build_vocab_from_iterator(yield_tokens(total_data_iter), min_freq=10, specials=['<unk>'])


import pickle
with open('./vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

vocab.set_default_index(0)
def collate_func(batch):
    target = []
    token_index = []
    max_length = 64

    for i, (label, comment) in enumerate(batch):
        token_idx = vocab(comment)
        token_idx = token_idx[:max_length]
        token_idx += [0] * (max_length - len(token_idx))
        token_index.append(token_idx)

        if label == 0:
            target.append(0)
        else:
            target.append(1)

    return (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32))

def train(train_data_loader,
          eval_data_loader,
          model,
          optimizer,
          num_epoch,
          log_step_interval,
          Model_Type,
          save_path,
          add_chart_data,
          resume=""
          ):

    start_epoch = 0
    min_acc = 0.0

    loss_train = []
    acc_train = []
    total, rights, loss_count = 0.0, 0.0, 0.0

    import torch.nn.functional as F
    criterion = nn.CrossEntropyLoss()
    for epoch_index in range(start_epoch, num_epoch):
        print(f"epoch: {epoch_index}")
        ema_loss = 0.
        num_batchs = len(train_data_loader)
        for batch_index, (target, index_token) in enumerate(train_data_loader):


            step = num_batchs * (epoch_index) + batch_index + 1
            logits = model(index_token)

            optimizer.zero_grad()

            result = torch.argmax(logits, dim=-1)
            rights += (result == target).sum().item() / target.shape[0]

            a = torch.sigmoid(logits)
            b = F.one_hot(target, num_classes=2).to(torch.float32)
            loss = F.binary_cross_entropy(a, b)


            ema_loss = 0.9 * ema_loss + 0.1 * loss
            loss_count += ema_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            total += 1.0
            if step % add_chart_data == 0:
                acc_train.append(rights / total)
                loss_train.append(loss_count / total)
                total, rights, loss_count = 0.0, 0.0, 0.0
            if step % log_step_interval == 0:
                logging.warning(f"epoch_index: {epoch_index}, batch_index: {batch_index}, ema_loss: {ema_loss}")
        logging.warning("start to do evaluation..")
        model.eval()
        ema_eval_loss = 0
        total_acc_account = 0
        total_account = 0

        for eval_batch_index, (eval_target, eval_index_token) in enumerate(eval_data_loader):
            total_account += eval_target.shape[0]
            eval_logits = model(eval_index_token)

            result = torch.argmax(eval_logits, dim=-1)
            total_acc_account += (result == eval_target).sum().item()

            eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits), F.one_hot(eval_target, num_classes=2).to(torch.float32))

            ema_eval_loss = 0.9 * ema_eval_loss + 0.1 * eval_bce_loss

        aver_eval_accuracy = total_acc_account / total_account
        logging.warning(f"eval_ema_loss: {ema_eval_loss}, eval_accuracy: {aver_eval_accuracy}")
        model.train()
        if aver_eval_accuracy > min_acc:
            os.makedirs(save_path, exist_ok=True)
            if os.path.exists(os.path.join(save_path, f'best_model_{Model_Type}_{num_epoch}_{format(min_acc, "0.5f")}.pt')):
                os.remove(os.path.join(save_path, f'best_model_{Model_Type}_{num_epoch}_{format(min_acc, "0.5f")}.pt'))
            save_file = os.path.join(save_path, f"best_model_{Model_Type}_{num_epoch}_{format(aver_eval_accuracy, '0.5f')}.pt")
            torch.save({
                'epoch': epoch_index,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_file)

            logging.warning(f"checkpoint has been saved in {save_file}")

            min_acc = aver_eval_accuracy
    loss_train_list = [loss.item() for loss in loss_train]
    plt.plot(loss_train_list, label='Train Loss')

    plt.plot(acc_train, label='Train Accuracy')

    plt.legend(loc='best')
    plt.xlabel('Batch')
    plt.ylabel('Value')
    plt.title(f'Comparison chart of accuracy and loss')
    plt.savefig(os.path.join("../Charts", f'plot_{Model_Type}_{min_acc}_{add_chart_data}.png'))
    plt.show()
    plt.close()
if __name__ == "__main__":
    myModel = CNN_LSTM()
    optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)
    train_data_iter = CustomDataset("train")


    train_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(train_data_iter),
                                                    batch_size=32,
                                                    collate_fn=collate_func,
                                                    shuffle=True)

    train_dataset_size = len(train_data_loader)
    eval_data_iter = CustomDataset("val")
    eval_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(eval_data_iter),
                                                   batch_size=8,
                                                   collate_fn=collate_func)
    eval_dataset_size = len(eval_data_loader)
    resume_path = ""
    the_save_path = ""

    train(train_data_loader,
          eval_data_loader,
          myModel,
          optimizer,
          num_epoch=4,
          log_step_interval=5,
          Model_Type="CNN_LSTM",
          save_path = "../my_model/CNN_LSTM",
          add_chart_data = 10,
          resume=resume_path)

