import torch
import torch.nn as nn
import pickle
from torchtext.data import get_tokenizer

import time
VOCAB_SIZE = 31650

class GCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, num_class=2):
        super(GCNN,self).__init__()

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.conv_A_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7, padding=56)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7, padding=56)

        self.conv_A_2 = nn.Conv1d(64, 64, 15, stride=7)
        self.conv_B_2 = nn.Conv1d(64, 64, 15, stride=7)

        self.output_linear1 = nn.Linear(64, 128)
        self.output_linear2 = nn.Linear(128, num_class)


    def forward(self, word_index):
        word_embedding = self.embedding_table(word_index)

        word_embedding = word_embedding.transpose(1, 2)
        A = self.conv_A_1(word_embedding)

        B = self.conv_B_1(word_embedding)

        H = A * torch.sigmoid(B)
        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B)
        pool_output = torch.mean(H, dim=-1)

        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output)

        return logits

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

        output = self.fc(out[:, -1, :])
        return output

class SimpleTextClassficationModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, num_class=2):
        super(SimpleTextClassficationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, token_index):
        embedded = self.embedding(token_index)

        logits = self.fc(embedded)
        return logits



checkpoint = torch.load("./best_model_CNN_LSTM_4_0.98971.pt")
text_model = CNN_LSTM()


# checkpoint = torch.load("./best_model_STC_4_0.98264.pt")
# text_model = SimpleTextClassficationModel()
#
# checkpoint = torch.load("./best_model_GCNN_10_0.99273.pt")
# text_model = SimpleTextClassficationModel()
#
#
# text_model.load_state_dict(checkpoint['model_state_dict'])
# text_model.eval()


vocab = None
with open('./vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
def preprocess_text(text):
    file = open('./stopwords_English.txt', 'r', encoding='utf-8')
    stop_words = [i.strip() for i in file.readlines()]
    file.close()
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(text)
    tokens = [i for i in tokens if not i in stop_words]

    token_indices = [[vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]]

    return torch.tensor(token_indices)
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

class imageModel(nn.Module):
    def __init__(self, num=2):
        super(imageModel, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.elu = nn.ELU()

        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)

        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        self.f6 = nn.Linear(4608, 2048)
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, num)

    def forward(self, x):
        x = self.elu(self.c1(x))
        x = self.elu(self.c2(x))
        x = self.s2(x)
        x = self.elu(self.c3(x))
        x = self.s3(x)
        x = self.elu(self.c4(x))
        x = self.elu(self.c5(x))
        x = self.s5(x)

        x = self.flatten(x)

        x = self.f6(x)
        x = F.dropout(x, p=0.5, training=True)
        x = self.f7(x)
        x = F.dropout(x, p=0.5, training=True)
        x = self.f8(x)
        x = F.dropout(x, p=0.5, training=True)

        x = self.f9(x)
        return x
class ImageClassifier:
    def __init__(self, model_path):
        self.model = imageModel(num=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            self.normalize
        ])

    def predict(self, theImage):
        image_pil = Image.open(theImage)
        image = self.transform(image_pil)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            output = self.model(image)
        return output
image_classifier = ImageClassifier("./best_model_20_0.970_.pkl")



# ----------- Text model reasoning time -----------
model_dict = {
    CNN_LSTM: ("CNN_LSTM", "./best_model_CNN_LSTM_4_0.98971.pt"),
    SimpleTextClassficationModel: ("STC (SimpleTextClassifier)", "./best_model_STC_4_0.98264.pt"),
    GCNN: ("GCNN", "./best_model_GCNN_10_0.99273.pt"),
}

text_input = preprocess_text("Hello, dear customer!Our store is having a big weekend promotion. If you spend over 1,000 US dollars, you can enjoy a 50% discount and receive many exquisite gifts.Additionally, there will be even bigger discounts at the end of this month. If you are interested, please call 010-5555-7777.We look forward to your visit.")

for model_class, (model_name, ckpt_path) in model_dict.items():
    checkpoint = torch.load(ckpt_path)
    text_model = model_class()
    text_model.eval()


    start = time.time()
    with torch.no_grad():
        output_text_model = text_model(text_input)
    end = time.time()
    probability_text = torch.sigmoid(output_text_model)
    print(f"[{model_name}] Classification Time: {end - start:.6f} s, \nThe probability that it is spam is {probability_text[0][1]}\n")

# ----------- image model reasoning time -----------
image_path = "./test_image.jpg"  #
start = time.time()
output_image_model= image_classifier.predict(image_path)
end = time.time()
probability_image = torch.sigmoid(output_image_model)
print(f"[Image Model] Classification Time: {end - start:.6f} seconds,  \nThe probability that it is spam is {probability_image[0][1]}")


