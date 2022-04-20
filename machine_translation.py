import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import spacy
import nltk
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchtext.data import Field, BucketIterator,TabularDataset
torch.cuda.empty_cache()
# python -m spacy download en --quiet
# python -m spacy download fr --quiet
punct = '''!()-[]{};:'"\,<>./?@#$%^&*_~,'''
english=spacy.load('en_core_web_sm')
french=spacy.load('fr_core_news_sm')
# punct = '''!()-[]{};:'"\,<>./?@#$%^&*_~,'''
def RemovePunct(text):
  data=[]
  for i in text:
    for j in i:
      if j in punct:
        i=i.replace(j," ")
    data.append(i)
  return data

with open("ted-talks-corpus/train.en","r") as file:
  english_train=file.readlines()
  english_train=RemovePunct(english_train)

with open("ted-talks-corpus/train.fr","r") as file:
  french_train=file.readlines()
  french_train=RemovePunct(french_train)


with open("ted-talks-corpus/test.en","r") as file:
  english_test=file.readlines()
  english_test=RemovePunct(english_test)

with open("ted-talks-corpus/test.fr","r") as file:
  french_test=file.readlines()
  french_test=RemovePunct(french_test)
  
with open("ted-talks-corpus/dev.en","r") as file:
  english_dev=file.readlines()
  english_dev=RemovePunct(english_dev)

with open("ted-talks-corpus/dev.fr","r") as file:
  french_dev=file.readlines()
  french_dev=RemovePunct(french_dev)
  
train=pd.DataFrame({'srg':english_train,'trg':french_train})
test=pd.DataFrame({'srg':english_test,'trg':french_test})
dev=pd.DataFrame({'srg':english_dev,'trg':french_dev})
train.to_csv("train.csv",index=False)
dev.to_csv("dev.csv",index=False)
test.to_csv("test.csv",index=False)


batch_size=16

def english_tokenizer(text):
      return  [token.text for token in english.tokenizer(text)]
  #  return text.split(" ")

def french_tokenizer(text):
  return  [token.text for token in french.tokenizer(text)]


english1 = Field(tokenize=english_tokenizer,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")

french1 = Field(tokenize=french_tokenizer,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")


train_data,valid_data,test_data=TabularDataset.splits(
    path="./",
    train="train.csv",validation="dev.csv",test="test.csv",format="csv",
    fields=[('src',english1),('trg',french1)]
    ,skip_header=True
)

english1.build_vocab(train_data)
# train_data.fields
french1.build_vocab(train_data)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                      batch_size = batch_size, 
                                                                      sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.src),
                                                                      device = device)


class Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,dropout):
        super(Encoder,self).__init__()
        self.vocab_size=vocab_size
        self.num_layers=1
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.dropout=nn.Dropout(dropout)
        self.embedding=nn.Embedding(self.vocab_size,self.embedding_size)
        self.lstm=nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers,dropout=dropout)


    def forward(self,input):
      # print("enter")
        embedding=self.dropout(self.embedding(input))
        out,(hidden_state,cell_state)=self.lstm(embedding)
        return hidden_state,cell_state
    
vocab_size=len(english1.vocab.itos)
embedding_size=300
hidden_size=512
# print(len(english1.vocab.itos))
dropout=0.5

encoder_lstm=Encoder(vocab_size,embedding_size,hidden_size,dropout).to(device)

print(encoder_lstm)

class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,dropout,output_size):
        super(Decoder,self).__init__()
        self.dropout=nn.Dropout(dropout)

        self.embedding=nn.Embedding(input_size,embedding_size)

        self.lstm=nn.LSTM(embedding_size,hidden_size,1,dropout=dropout)

        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,input,hidden_state,cell_state):
        input=input.unsqueeze(0)
        embedding=self.dropout(self.embedding(input))

        out,(hidden_state,cell_state)=self.lstm(embedding,(hidden_state,cell_state))

        prediction=self.fc(out)

        predictions=prediction.squeeze(0)
        return predictions,hidden_state,cell_state
    
vocab_size_f=len(french1.vocab.itos)
decoder_lstm=Decoder(vocab_size_f,embedding_size,hidden_size,dropout,vocab_size_f).to(device)

import random
class Seq2Seq(nn.Module):
    def __init__(self, Encoder_LSTM, Decoder_LSTM):
        super(Seq2Seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM

    def forward(self, source, target, tfr=0.5):
       
        batch_size = source.shape[1]

        
        target_len = target.shape[0]
        target_vocab_size = len(french1.vocab.itos)
        
      
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
       
        hidden_state, cell_state = self.Encoder_LSTM(source)

       
        x = target[0] # Trigger token <SOS>

        for i in range(1, target_len):
       
        # print()
            output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
            # print(output.shape)
            outputs[i] = output
            best_guess = output.argmax(1) 
            x = target[i] if random.random() < tfr else best_guess 
       
        return outputs
    
model=Seq2Seq(encoder_lstm,decoder_lstm).to(device)
optimizer=optim.Adam(model.parameters(),lr=1e-3)

pad_ids=french1.vocab.stoi["<pad>"]

criterion=nn.CrossEntropyLoss(ignore_index=pad_ids)

print(model)

# for name, param in model.named_parameters():
#     nn.init.uniform_(param.data, -0.08, 0.08)
    

# epochs=20
# best_epoch=-1
# best_loss=float('inf')
# # sent=train[0]

# for epo in range(epochs):
#   print(f"Epoch {epo} is running.... \n")
#   # model.eval()
#   epochs_loss_training=0.0
#   epochs_loss_validating=0.0

#   model.train()
#   for idx,batch in enumerate(train_iterator):
#     input=batch.src.to(device)
#     target=batch.trg.to(device)
#     # print(input.shape,target.shape)
#     output=model.forward(input,target)
#     output=output[1:].reshape(-1,output.shape[2])
#     target=target[1:].reshape(-1)

    

#     loss=criterion(output,target)

#     loss.backward()

#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

#     optimizer.step()
#     epochs_loss_training+=loss.item()
#     # print(loss.item())
#   print("Average Training loss...")
#   print(epochs_loss_training/len(train_iterator))

#   model.eval()
#   for idx,bat in enumerate(valid_iterator):
#     input=bat.src.to(device)
#     target=bat.trg.to(device)
   
#     output=model(input,target)
#     output=output[1:].reshape(-1,output.shape[2])
#     target=target[1:].reshape(-1)

#     loss=criterion(output,target)
#     epochs_loss_validating+=loss.item()

#   if best_loss > epochs_loss_validating:
#     best_loss=epochs_loss_validating
#     state={
#         "model":model.state_dict(),
#         "optimizer":optimizer.state_dict(),
#         "rng_state":torch.get_rng_state(),
#     }
#     torch.save(state,'MT-1')
#     best_epoch=epo
#   print("Average Validation loss...")
#   print(epochs_loss_validating/len(valid_iterator))
  
  #print("epoch_loss- {}".format(loss.item()))

# state={
#         "model":model.state_dict(),
#         "optimizer":optimizer.state_dict(),
#         "rng_state":torch.get_rng_state(),
#     }
# torch.save(state,'MT-2')
# print(epochs_loss_validating)
model.load_state_dict(torch.load("MT2-1")['model'])
import math
def Transalate(model,sent,english1,french1,device,max_len=50):
    model.eval()
    if type(sent)==str:
      tokens=[token.text.lower() for token in english(sent)]
    else:
      tokens = [token.lower() for token in sent]
    tokens.insert(0, english1.init_token)
    tokens.append(english1.eos_token)
    text_to_indices = [english1.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [french1.vocab.stoi["<sos>"]]
    len_poss=0
    for _ in range(max_len):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        sum=0
        with torch.no_grad():
            len_poss+=1
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()
            # print(output)
            a=nn.Softmax(dim=1)
            p=a(output)
            # print(p)
            l=torch.max(p,dim=1)
            # print((l[0][0]).cpu().numpy())
            sum+=math.log(l[0][0].cpu().numpy())
            # print(l.cpu().numpy())
            # print(output.argmax())
        
        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == french1.vocab.stoi["<eos>"]:
            # print(math.exp(-1*(sum)/len_poss))
            break

    translated_sentence = [french1.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]
  
# from torchtext.data.metrics import bleu_score
# for i in english_test:
# import os

sentence=input("enter your sentence")

   
print(" ".join(Transalate(model,sentence,english1,french1,device)))
