# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import HAN_Model
from data import IMDB_Data
import numpy as np
from tqdm import tqdm
import config as argumentparser
from transformers import BertTokenizer
import os

from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW


config = argumentparser.ArgumentParser()
torch.manual_seed(config.seed)

if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)

def get_test_result(data_iter,data_set):
    # 生成测试结果
    model.eval()
    true_sample_num = 0
    for data, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        if config.cuda and torch.cuda.is_available():
            out = model(data, gpu=True)
        else:
            out = model(data)
        true_sample_num += np.sum((torch.argmax(out, 1) == label).cpu().numpy())
    acc = true_sample_num / data_set.__len__()
    return acc



model_name_or_path = os.path.abspath(".") + "/prev_trained_model/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
training_set = IMDB_Data("toutiao_article_score_train.txt",min_count=config.min_count,
                         max_sentence_length = config.max_sentence_length,batch_size=config.batch_size,tokenizer=tokenizer)
training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=0)
test_set = IMDB_Data("toutiao_article_score_test.txt",min_count=config.min_count,
                         max_sentence_length = config.max_sentence_length,batch_size=config.batch_size,tokenizer=tokenizer)
test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=0)
if config.cuda and torch.cuda.is_available():
    training_set.weight = training_set.weight.cuda()
model = HAN_Model( gru_size = config.gru_size,class_num=config.class_num)
if config.cuda and torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss = -1

if config.max_steps > 0:
    t_total = config.max_steps
    config.num_train_epochs = config.max_steps // (len(training_iter) // config.gradient_accumulation_steps) + 1
else:
    t_total = len(training_iter) // config.gradient_accumulation_steps * config.epoch
# Prepare optimizer and schedule (linear warmup and decay)


no_decay = ["bias", "LayerNorm.weight"]
bert_param_optimizer = list(model.bert.named_parameters())

word_gru_param_optimizer = list(model.word_gru.named_parameters())
word_context_param_optimizer = list(model.word_context.named_parameters())
word_dense_param_optimizer = list(model.word_dense.named_parameters())
sentence_gru_param_optimizer = list(model.sentence_gru.named_parameters())
sentence_context_param_optimizer = list(model.sentence_context.named_parameters())
sentence_dense_param_optimizer = list(model.sentence_dense.named_parameters())


linear_param_optimizer = list(model.classifier.named_parameters())


optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.learning_rate},
    {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.learning_rate},



    {'params': [p for n, p in word_gru_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in word_gru_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.crf_learning_rate},
    {'params': [p for n, p in word_context_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in word_context_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.crf_learning_rate},
    {'params': [p for n, p in word_dense_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in word_dense_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.crf_learning_rate},
    {'params': [p for n, p in sentence_gru_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in sentence_gru_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.crf_learning_rate},
    {'params': [p for n, p in sentence_context_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in sentence_context_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.crf_learning_rate},
    {'params': [p for n, p in sentence_dense_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in sentence_dense_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.crf_learning_rate},



    {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.crf_learning_rate}
]


config.warmup_steps = int(t_total * config.warmup_proportion)
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=t_total)




for epoch in range(config.epoch):
    model.train()
    process_bar = tqdm(training_iter)
    for data,maks,segment,label in process_bar:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        label = torch.autograd.Variable(label).squeeze()
        if config.cuda and torch.cuda.is_available():
            out = model(data,maks,segment,gpu=True)
        else:
            out = model(data,maks,segment)
        loss_now = criterion(out, autograd.Variable(label.long()))
        if loss == -1:
            loss = loss_now.data.item()
        else:
            loss = 0.95*loss+0.05*loss_now.data.item()
        process_bar.set_postfix(loss=loss_now.data.item())
        process_bar.update()
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()
    test_acc = get_test_result(test_iter, test_set)
    print("The test acc is: %.5f" % test_acc)