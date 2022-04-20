from torchtext.data import Field, BucketIterator, TabularDataset
from lib2to3.pgen2.tokenize import tokenize
import numpy as np
import pandas as pd
import os
from zipfile import ZipFile
import zipfile

import spacy

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

import math
import time

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')


def Decoder(data):
    dec = []
    for i in data:
        dec.append(i.decode("UTF-8"))
    return dec


archive = zipfile.ZipFile("europarl-corpus.zip", "r")
with ZipFile("europarl-corpus.zip", 'r') as zip:
    listoffiles = zip.namelist()
    with zip.open("europarl-corpus/dev.europarl", "r") as file:
        dev = file.readlines()
    file.close()
    with zip.open("europarl-corpus/train.europarl", "r") as file:
        train = file.readlines()
    file.close()
    with zip.open("europarl-corpus/test.europarl", "r") as file:
        test = file.readlines()
    file.close()


# train = Decoder(train)
# valid = Decoder(dev)
# test=Decoder(test)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = train = Decoder(train)
b = dev = Decoder(dev)
c = test = Decoder(test)
train = pd.DataFrame({'srg': train})
dev = pd.DataFrame({'srg': dev})
test = pd.DataFrame({'srg': test})
train.to_csv("train_l.csv", index=False)
dev.to_csv("dev_l.csv", index=False)
test.to_csv("test_l.csv", index=False)

english = spacy.load('en_core_web_sm')


def english_tokenizer(text):
    return [token.text for token in english.tokenizer(text)]


english1 = Field(tokenize=english_tokenizer,
                 lower=True,
                 init_token="<sos>",
                 eos_token="<eos>")


train_data, valid_data, test_data = TabularDataset.splits(
    path="./",
    train="train_l.csv", validation="dev_l.csv", test="test_l.csv", format="csv",
    fields=[('src', english1)], skip_header=True
)

english1.build_vocab(train_data)


# def yield_tokens(train):
#     for line in train:
#         yield tokenizer(line)


# vocab = torchtext.vocab.build_vocab_from_iterator(
# yield_tokens(train+valid), specials=["<sos>", "<eos>"])


def get_data(dataset, batch_size):
    data = []
    # min1=0
    for line in dataset:
        l = ["<sos>"]
        l += (english_tokenizer(line))
        l.append("<eos>")
        tokens = [english1.vocab.stoi[token] for token in l]
        data.extend(tokens)
    data = torch.LongTensor(data).to(device)
    batches = data.shape[0]//batch_size
    data = data.narrow(0, 0, batches*batch_size)
    data = data.view(batch_size, -1)
    return data


batch_size = 16


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.n_layers = 1
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size,
                             self.hidden_dim).to(device)
        cell = torch.zeros(self.n_layers, batch_size,
                           self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, input, hidden):
        # input = [batch size, seq len]
        # hidden = [n layers, batch size, hidden dim]
        embedding = self.dropout(self.embedding(input))
        # embedding = [batch size, seq len, embedding dim]
        # print(hidden[0].shape, hidden[1].shape)
        output, hidden = self.lstm(embedding, hidden)
        # output = [batch size, seq len, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        output = self.dropout(output)
        prediction = self.fc(output)
        # prediction = [batch size, seq len, vocab size]
        return prediction, hidden


learning_rate = 0.001
model = Model(len(english1.vocab.itos), 300, 512, 0.5)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


def get_batch(data, max_seq_len, tokens, offset):
    seq_len = min(max_seq_len, tokens-offset-1)
    input = data[:, offset:offset+seq_len]
    target = data[:, offset+1:offset+1+seq_len]
    return input, target, seq_len


def trainmodel(model, data, optimizer, criterion, batch_size, max_seq_len, device):
    epoch_loss = 0
    model.train()
    hidden = model.init_hidden(batch_size, device)
    # print(hidden)
    for offset in range(0, data.shape[-1]-1, max_seq_len):
        optimizer.zero_grad()
        input, target, seq_len = get_batch(
            data, max_seq_len, data.shape[-1], offset)
        input, target = input.to(device), target.to(device)
        batch_size, seq_len = input.shape
        # print(hidden[0].shape)
        # print(input.shape)
        hidden = model.detach_hidden(hidden)
        prediction, hidden = model(input, hidden)
        # hidden,cell=hidden
        prediction = prediction.reshape(batch_size*seq_len, -1)
        target = target.reshape(-1)

        loss = criterion(prediction, target)

        loss.backward()
        epoch_loss += loss.item()*seq_len

        optimizer.step()
    return epoch_loss/data.shape[-1]


def evalutemodel(model, data, criterion, batch_size, max_seq_len, device):
    epoch_loss = 0
    model.train()
    hidden, cell = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for offset in range(0, data.shape[-1], max_seq_len):
            input, target, seq_len = get_batch(
                data, max_seq_len, data.shape[-1], offset)
            input, target = input.to(device), target.to(device)
            batch_size, seq_len = input.shape

            prediction, hidden = model(input, (hidden, cell))
            hidden, cell = hidden
            prediction = prediction.reshape(batch_size*seq_len, -1)
            target = target.reshape(-1)
            loss = criterion(prediction, target)
            epoch_loss += loss.item()*seq_len

    return epoch_loss/data.shape[-1]


# train_data = get_data(a,batch_size)
# valid_data = get_data(b,batch_size)
# # train_data=get_data(my_test_data,vocab,batch_size)
# best_loss = float('inf')
# max_seq_len = 42
# for epoch in range(100):
#     print(f'Epoch {epoch} running....')
#     train_loss = trainmodel(model, train_data, optimizer,
#                             criterion, batch_size, max_seq_len, device)
#     print(f'training_loss\n {train_loss}')
#     # print(train_loss)
#     valid_loss = evalutemodel(
#         model, valid_data, criterion, batch_size, max_seq_len, device)
#     # print(valid_loss)
#     print(f'validatin loss \n {valid_loss}')
#     if epoch % 10 == 0:
#         print("model saved")
#         torch.save(model.state_dict(), f'lm{epoch}')
#     if valid_loss < best_loss:
#         best_loss = valid_loss
#         torch.save(model.state_dict(), 'lm')

model.load_state_dict(torch.load('lm_f'))


def generate(prompt, n_gen_tokens, temperature, model, tokenizer, device, seed=None):
    if seed is not None:
        torch.manual_seed(0)
    model.eval()
    tokens = english_tokenizer(prompt)
    indices = [english1.vocab.stoi[t] for t in tokens]
    # print(indices)
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    itos = english1.vocab.itos
    length = 0
    sum = 0
    with torch.no_grad():
        length += 1
        for i in range(n_gen_tokens):
            input = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(input, hidden)
            # print(prediction.shape)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            # print(probs.shape)
           
            prediction = torch.multinomial(probs, num_samples=1).item()
            # print(prediction)
            if itos[prediction] == "<eos>":
                # print(math.exp(-1*(sum/length)))
                break
            sum += math.log(torch.max(probs).item())
            length += 1
            indices.append(prediction)
            

    # print(indices)

    tokens = [itos[i] for i in indices]
    return " ".join(tokens), math.exp(-1*(sum/length))


avg_score = 0
prompt = 'the'
preplexity = []


# print(generation)
try:
    os.remove("train_pre.txt")
except:
    pass




def Prepelexity(sent,model):
    tokens=english.tokenizer(sent)
    temperature=0.5

    indices = [english1.vocab.stoi[tokens[0]]]
    # print(indices)
    
        # print(indices)
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    itos = english1.vocab.itos
    size=0
    probabi=1
    sum=0
    with torch.no_grad():
        for i in range(1,len(tokens)):
            input = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(input, hidden)
                    # print(prediction.shape)
            index=english1.vocab.stoi[tokens[i]]
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1) 
            #  print(probs[0:index])
            # print(probs)
            # print(probs[0,index].item())
            probabi*=probs[0,index].item()
            indices.append(index)
            sum+=math.log(probs[0,index].item())

    
    return math.exp(-1*(sum/len(tokens))),probabi

# for i in a:
#     try:
#         # sent, pre = generate(prompt, n_gen_tokens,
#         #                      temperature, model, tokenizer, device, seed)
#         pre=Prepelexity(i,model)
#         preplexity.append([i, pre])
#         avg_score += pre
#     except:
#         print(0.000)
        

# with open("train_pre.txt", "a") as file:
#     file.write(str(avg_score/len(a)))
#     file.write("\n")
#     for i in preplexity:
#         string = str(i[0])+"  "+str(i[1])
#         file.write(string)
#         file.write("\n")
sentence=input("enter your sentence : ")

pre,prob=Prepelexity(sentence,model)
print("prelexity of sentence is:",pre)
print("probability of sentence is:",prob)