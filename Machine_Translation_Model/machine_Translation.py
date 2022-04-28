import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim
from nltk.tokenize import word_tokenize
import os
import nltk
import codecs
import csv
from konlpy.tag import Okt
import torch.nn.functional as F
nltk.download('punkt')


#
with open("train_en_ko.csv") as csv_f:
    head = "\n".join([next(csv_f) for x in range(5)])
print(head)


class GRUMT(nn.Module):
    # GRU 기반 MT 클래스를 정의합니다. Pytorch는 모델을 구성할 때 반드시 nn.Module 클래스를 상속받은 후 이를 토대로 만듭니다.
    def __init__(self, input_size, hidden_size, output_size, max_length, device, dropout_p=0.1):
        # 클래스의 첫 시작인 함수입니다. 여기서 모델에 필요한 여러 변수들을 정의합니다.
        super(GRUMT, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        # Encoder 부분
        self.encoder_embedding = nn.Embedding(input_size, hidden_size)
        # <ToDo>: encoder를 GRU로 정의하세요.
        # self.encoder_rnn = None
        # self.encoder_rnn = nn.GRU(self.input_size,self.hidden_size,num_layers=1,bias=True,batch_first=True,bidirectional=True)
        self.encoder_rnn = nn.GRU(self.hidden_size, self.hidden_size)

        # Decoder 부분
        # <ToDo>: decoder를 GRU로 정의하세요.
        # self.decoder_rnn = None
        # self.decoder_rnn = nn.GRU(self.output_size,self.hidden_size,num_layers=1,bias=True,batch_first=True,bidirectional=True) drop_last=True
        self.decoder_rnn = nn.GRU(self.hidden_size, self.hidden_size)
        self.decoder_embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.loss = nn.NLLLoss()

    def _encoder(self, input_tensor, input_length):
        # forward 함수 중 첫 번째 부분인 encoder에 대한 함수입니다.
        encoder_hidden = self._init_hidden()

        encoder_outputs = torch.zeros(self.max_length, self.hidden_size, device=self.device)

        # input_tensor의 길이만큼 하나씩 GRU를 통과시키고 그 결과를 저장합니다.
        for idx in range(input_length):
            input_tensor_step = input_tensor[idx]
            embedded = self.encoder_embedding(input_tensor_step).view(1, 1, -1)
            encoder_output, encoder_hidden = self.encoder_rnn(embedded, encoder_hidden)
            encoder_outputs[idx] = encoder_output[0, 0]

        return encoder_outputs, encoder_hidden

    def _decoder(self, target_tensor, target_length, encoder_hidden, encoder_outputs):
        # forward 함수 중 두 번째 부분인 decoder에 대한 함수입니다.
        # decoder의 입력은 특수 문자인 SOS입니다.
        decoder_input = torch.tensor([[SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden

        loss_sum = 0
        # 번역할 문장의 길이만큼 단어를 생성합니다.
        # 단어 생성은 주어진 언어 사전에 있는 단어 중 하나를 선택하는 classification 문제와 동일합니다.
        for di in range(target_length):
            embedded = self.decoder_embedding(decoder_input).view(1, 1, -1)
            embedded = self.dropout(embedded)

            # encoder의 결과와 decoder의 hidden을 결합하여 현재 생성할 단어에 영향을 많이 주는 attention을 구합니다.
            decoder_attention = F.softmax(self.attn(torch.cat((embedded[0], decoder_hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(decoder_attention.unsqueeze(0), encoder_outputs.unsqueeze(0))

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)

            output = F.relu(output)
            output, decoder_hidden = self.decoder_rnn(output, decoder_hidden)

            decoder_output = F.log_softmax(self.out(output[0]), dim=1)

            # decoder를 거쳐 나온 출력 중 가장 높은 값을 가지는 단어를 찾습니다.
            # 찾은 단어는 다음 반복문의 입력 단어가 됩니다.
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            target_output = torch.tensor([target_tensor[di]], device=self.device)

            # 그리고 그 단어와 실제 단어의 차이를 loss로 정의합니다.
            loss_sum += self.loss(decoder_output, target_output)
            if decoder_input.item() == EOS_token:
                break

        return loss_sum

    def forward(self, input_tensor, input_length, target_tensor, target_length):
        # 모델의 forward feed를 수행하는 함수입니다.
        # 영어 문장과 한국어 문장 두 개를 받아 영어 문장에서 한국어 문장을 만드는 seq2seq 모델입니다.
        # Encoder 파트
        encoder_outputs, encoder_hidden = self._encoder(input_tensor, input_length)

        # Decoder 파트
        loss_sum = self._decoder(target_tensor, target_length, encoder_hidden, encoder_outputs)

        return loss_sum

    def _init_hidden(self):
        # encoder와 decoder 둘 다 처음에 가지는 hidden입니다. 간단히 0으로 시작합니다.
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class LangDic:
    # 언어마다 단어 사전을 정의합니다.
    def __init__(self, name, tokenizer):
        # 클래스의 첫 시작인 함수입니다. 여기서 모델에 필요한 여러 변수들을 정의합니다.
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 2
        self.tokenizer = tokenizer

    def add_sentence(self, sentence):
        # 문장을 받아 문장에서 단어를 확인합니다.
        for word in self.tokenizer(sentence):
            self.add_word(word)

    def add_word(self, word):
        # 단어를 보고 그 단어가 사전에 존재하는지 아닌지를 살펴봅니다.
        # 존재하지 않는 경우 단어를 사전에 등록합니다.
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence2tensor(self, sentence, max_len):
        # 가지고 있는 사전을 바탕으로 문장을 tensor의 형태로 바꿉니다.
        # tensor 내 들어있는 값은 단어 index이며 이를 통해 모델의 임베딩에 입력으로 줄 수 있습니다.
        indexes = list()
        for word in self.tokenizer(sentence):
            try:
                indexes.append(self.word2index[word])
            except KeyError:
                indexes.append(UNK_token)

        indexes.append(EOS_token)
        len_sen = len(indexes)
        if len_sen > max_len:
            indexes = indexes[:max_len-1]
            indexes.append(EOS_token)
            len_sen = max_len

        index_tensor = torch.tensor(indexes)
        return index_tensor, len_sen

def make_dic(dataset_path):
    # 사전을 만드는 함수입니다.
    data_pairs = load_file(dataset_path)

    # 영어의 경우 NLTK tokenizer를
    # 한국어의 경우 Konlpy 내 Open Korean Text tokenizer를 이용합니다.
    eng_tokenizer = word_tokenize
    kor_tokenizer = Okt().morphs

    eng_dic = LangDic('en', eng_tokenizer)
    kor_dic = LangDic('ko', kor_tokenizer)

    for eng_sen, kor_sen in data_pairs:
        eng_dic.add_sentence(eng_sen)
        kor_dic.add_sentence(kor_sen)

    return eng_dic, kor_dic

def load_file(dataset_path):
    # 데이터를 읽는 함수입니다.
    data_pairs = list()
    # 데이터 파일의 내용을 불러와 영어 문장과 한국어 문장을 모아 리스트에 넣습니다.
    with codecs.open(dataset_path, "r", "utf-8") as csv_f:
        csv_reader = csv.reader(csv_f)
        for one_row in csv_reader:
            data_pairs.append(one_row)

    return data_pairs

class MTDataset(Dataset):
    # pytorch로 데이터를 불러오기 위해서 Dataset 클래스를 상속받아 새로운 클래스를 만듭니다.
    def __init__(self, data_pairs, eng_dic, kor_dic, max_len):
        super(MTDataset, self).__init__()

        # 데이터를 파일로부터 읽어 이를 전달 받습니다.
        self.max_len = max_len
        self.pair_data = list()

        # 데이터 내 문장을 미리 정의한 사전에 기반하여 tensor로 바꿉니다.
        for eng_sen, kor_sen in data_pairs:
            eng_sen_words, eng_sen_len = eng_dic.sentence2tensor(eng_sen, max_len)
            kor_sen_words, kor_sen_len = kor_dic.sentence2tensor(kor_sen, max_len)
            self.pair_data.append((eng_sen_words, eng_sen_len, kor_sen_words, kor_sen_len))

        self.data_len = len(self.pair_data)

    def __getitem__(self, idx):
        # idx번째 데이터를 반환합니다.
        eng_sen_words, eng_sen_len, kor_sen_words, kor_sen_len = self.pair_data[idx]

        return eng_sen_words, eng_sen_len, kor_sen_words, kor_sen_len

    def __len__(self):
        return self.data_len

def make_data_loader(dataset_path, eng_dic, kor_dic, max_len, batch_size):
    # DataLoader를 만들어서 데이터를 불러오도록 합니다.
    data_pairs = load_file(dataset_path)

    # 앞서 정의한 MTDataset 클래스에 해당 데이터를 넣습니다.
    ds = MTDataset(data_pairs, eng_dic, kor_dic, max_len)

    # 만들어진 MTDataset 클래스를 DataLoader에 넣고 batch 크기를 전달해줍니다.
    return DataLoader(ds, batch_size=batch_size)


def train(model, device, optimizer, train_loader, valid_loader, num_epochs):
    # 학습에 필요한 변수들을 기본적으로 정의합니다.
    running_loss = 0.0
    global_step = 0
    eval_every = 100

    # model에게 학습이 진행됨을 알려줍니다.
    model.train()
    # num_epochs만큼 epoch을 반복합니다.
    for epoch in range(num_epochs):
        # train_loader를 읽으면 정해진 데이터를 읽어옵니다.
        for input_tensor, input_length, target_tensor, target_length in train_loader:
            # 데이터를 GPU로 옮깁니다.
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            # model을 함수처럼 호출하면 model에서 정의한 forward 함수가 실행됩니다.
            # 즉, 데이터를 모델에 집어넣어 forward방향으로 흐른 후 그 결과를 받습니다.
            loss_sum = model(input_tensor, input_length, target_tensor, target_length)

            # 최적화 수행
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            running_loss += loss_sum.item() / target_length.item()
            global_step += 1

            if global_step % eval_every == 0:
                # 100번에 한 번으로 validation 데이터를 이용하여 성능을 검증합니다.
                print_loss_avg = running_loss / eval_every

                average_valid_loss = evaluate(model, device, valid_loader)

                # 검증이 끝난 후 다시 모델에게 학습을 준비시킵니다.
                model.train()
                running_loss = 0.0

                # 결과 출력
                print('Epoch {}, Step {}, Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, global_step, print_loss_avg, average_valid_loss))


def evaluate(model, device, valid_loader):
    # 학습 중 모델을 평가합니다.
    # 모델에게 학습이 아닌 평가를 할 것이라고 알립니다.
    model.eval()
    valid_running_loss = 0.0

    # 학습이 아니기에 최적화를 하지 않는다는 환경을 설정합니다.
    with torch.no_grad():
        # validation 데이터를 읽습니다.
        for input_tensor, input_length, target_tensor, target_length in valid_loader:
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            # model을 함수처럼 호출하면 model에서 정의한 forward 함수가 실행됩니다.
            # 즉, 데이터를 모델에 집어넣어 forward방향으로 흐른 후 그 결과를 받습니다.
            loss_sum = model(input_tensor, input_length, target_tensor, target_length)

            # validation 데이터의 loss, 즉 모델의 출력과 실제 데이터의 차이를 구합니다.
            valid_running_loss += loss_sum.item() / target_length.item()

    # 평균 loss를 계산하여 반환합니다.
    return valid_running_loss / len(valid_loader)


################################################################################
###############################  TEST  #########################################
################################################################################
on_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사전 내 특수 단어의 index를 각각 미리 정의합니다.
SOS_token = 0
EOS_token = 1
UNK_token = 2

# 데이터의 파일 정보
train_file_path = "./train_en_ko.csv"
valid_file_path = "./valid_en_ko.csv"

max_len = 10
hidden_size = 256

# 언어 별 사전 생성
eng_dic, kor_dic = make_dic(dataset_path=train_file_path)

#  train, validation 데이터 csv 파일을 읽어옵니다.
train_loader = make_data_loader(train_file_path, eng_dic, kor_dic, max_len, 1)

# <ToDo>: valid_dataset을 불러오세요.
# valid_loader = None
valid_loader = make_data_loader(valid_file_path, eng_dic, kor_dic, max_len, 1)

# 영어와 한국어 사전의 크기(단어 개수)를 가져옵니다.
eng_dic_size = eng_dic.n_words
kor_dic_size = kor_dic.n_words

# <ToDo>: GRUMT 클래스의 인스턴스를 만드세요. 인스턴스 생성 시 필요한 parameter를 전달해주세요.
# model = GRUMT(None).to(on_device) # 'input_size', 'hidden_size', 'output_size', 'max_length', and 'device'
model = GRUMT(input_size=eng_dic_size, hidden_size=hidden_size, output_size=kor_dic_size, max_length=max_len,device=on_device).to(on_device)

# Adam optimizier를 사용합니다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# <ToDo>: 학습을 위해 train 함수의 적절한 parameter를 전달해주세요.
# train(None) # model, device, optimizer, train_loader, valid_loader, num_epochs
train(model,on_device, optimizer, train_loader, valid_loader, num_epochs=12)