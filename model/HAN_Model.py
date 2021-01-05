# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from transformers.modeling_bert import BertModel

class HAN_Model(nn.Module):
    def __init__(self, gru_size, class_num):
        super(HAN_Model, self).__init__()

        self.bert = BertModel.from_pretrained("./prev_trained_model/bert-base")
        self.hidden_size = 768

        self.word_gru = nn.GRU(input_size=self.hidden_size,hidden_size=gru_size,num_layers=1,
                               bidirectional=True,batch_first=True)
        self.word_context = nn.Parameter(torch.Tensor(2*gru_size, 1),requires_grad=True)
        self.word_dense = nn.Linear(2*gru_size,2*gru_size)

        self.sentence_gru = nn.GRU(input_size=2*gru_size,hidden_size=gru_size,num_layers=1,
                               bidirectional=True,batch_first=True)
        self.sentence_context = nn.Parameter(torch.Tensor(2*gru_size, 1),requires_grad=True)
        self.sentence_dense = nn.Linear(2*gru_size,2*gru_size)
        self.classifier = nn.Linear(2*gru_size, class_num)

    def forward(self, x, mask, segment,gpu=False):
        sentence_num = x.shape[1] #x:b,p,s
        sentence_length = x.shape[2]
        batch = x.shape[0]
        x_embedding = torch.zeros([batch, sentence_num, sentence_length, self.hidden_size])

        if gpu:
            x_embedding = x_embedding.cuda()

        for i in range(sentence_num):
            outputs = self.bert(input_ids=x[:,i,...], attention_mask=mask[:,i,...], token_type_ids=segment[:,i,...]) # b,s,e
            x_embedding[:, i, ...] = outputs[0]

        x_embedding = x_embedding.view([-1, sentence_length, self.hidden_size]) #b,p,s,h->b*p,s,h
        mask = mask.view([-1, sentence_length]) #mask: b*p,s

        word_outputs, word_hidden = self.word_gru(x_embedding) #word_outputs:b*p,s,h
        attention_word_outputs = torch.tanh(self.word_dense(word_outputs))
        weights = torch.matmul(attention_word_outputs,self.word_context) #weights: b*p,s,1
        weights = F.softmax(weights,dim=1)

        mask = mask.unsqueeze(2)
        if gpu:
            weights = torch.where(mask!=0,weights,torch.full_like(mask,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(mask != 0, weights, torch.full_like(mask, 0, dtype=torch.float))

        weights = weights/(torch.sum(weights,dim=1).unsqueeze(1)+1e-4)

        sentence_vector = torch.sum(word_outputs*weights,dim=1).view([-1,sentence_num,word_outputs.shape[-1]])
        sentence_outputs, sentence_hidden = self.sentence_gru(sentence_vector)
        attention_sentence_outputs = torch.tanh(self.sentence_dense(sentence_outputs))
        weights = torch.matmul(attention_sentence_outputs,self.sentence_context)
        weights = F.softmax(weights,dim=1)

        mask = mask.view(-1, sentence_num, mask.shape[1])
        mask = torch.sum(mask, dim=2).unsqueeze(2)
        if gpu:
            weights = torch.where(mask != 0, weights, torch.full_like(mask,0,dtype=torch.float).cuda())
        else:
            weights = torch.where(mask != 0, weights, torch.full_like(mask, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights,dim=1).unsqueeze(1)+1e-4)
        document_vector = torch.sum(sentence_outputs*weights,dim=1)
        output = self.classifier(document_vector)
        return output

if __name__=="__main__":
    han_model = HAN_Model(vocab_size=30000,embedding_size=200,gru_size=50,class_num=4)
    x = torch.Tensor(np.zeros([64,50,100])).long()
    x[0][0][0:10] = 1
    output = han_model(x)
    print (output.shape)