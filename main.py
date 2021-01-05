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
import time

from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from tools.common import init_logger, logger

config = argumentparser.ArgumentParser()
torch.manual_seed(config.seed)

if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)

def get_test_result(data_iter,data_set):
    # 生成测试结果
    model.eval()
    true_sample_num = 0
    for data, maks, segment, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            maks = maks.cuda()
            segment = segment.cuda()

            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
            maks = torch.autograd.Variable(maks).long()
            segment = torch.autograd.Variable(segment).long()

        if config.cuda and torch.cuda.is_available():
            out = model(data,maks,segment, gpu=True)
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

time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
init_logger(log_file=config.output_dir + f'/{config.model_type}-{time_}.log')


model = HAN_Model( gru_size = config.gru_size,class_num=config.class_num)
if config.cuda and torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
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
# word_context_param_optimizer = list(model.word_context.named_parameters())
word_dense_param_optimizer = list(model.word_dense.named_parameters())
sentence_gru_param_optimizer = list(model.sentence_gru.named_parameters())
# sentence_context_param_optimizer = list(model.sentence_context.named_parameters())
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
     'lr': config.score_learning_rate},
    {'params': [p for n, p in word_dense_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in word_dense_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.score_learning_rate},
    {'params': [p for n, p in sentence_gru_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in sentence_gru_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.score_learning_rate},
    {'params': [p for n, p in sentence_dense_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in sentence_dense_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.score_learning_rate},



    {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay, 'lr': config.score_learning_rate},
    {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': config.score_learning_rate}
]


config.warmup_steps = int(t_total * config.warmup_proportion)
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=t_total)

if config.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16_opt_level)
# multi-gpu training (should be after apex fp16 initialization)


# Train!
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(training_iter))
logger.info("  Num Epochs = %d", config.epoch)
logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
            config.batch_size
            * config.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if config.local_rank != -1 else 1),
            )
logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
logger.info("  Total optimization steps = %d", t_total)
global_step = 0
steps_trained_in_current_epoch = 0
model.zero_grad()

for epoch in range(config.epoch):
    model.train()
    process_bar = tqdm(training_iter)
    for step, (data, maks, segment, label) in enumerate(process_bar):

        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            maks = maks.cuda()
            segment = segment.cuda()

            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
            maks = torch.autograd.Variable(maks).long()
            segment = torch.autograd.Variable(segment).long()

        label = torch.autograd.Variable(label).squeeze()
        if config.cuda and torch.cuda.is_available():
            out = model(data,maks,segment,gpu=True)
        else:
            out = model(data,maks,segment)

        logger.info("batch {0}, {1}".format(out.shape, label.shape))

        loss_now = criterion(out, autograd.Variable(label.long()))
        if loss == -1:
            loss = loss_now.data.item()
        else:
            loss = 0.95*loss+0.05*loss_now.data.item()

        process_bar.set_postfix(loss=loss_now.data.item())
        process_bar.update()

        if config.fp16:
            with amp.scale_loss(loss_now, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_now.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if config.local_rank in [-1, 0] and config.logging_steps > 0 and global_step % config.logging_steps == 0:
                # Log metrics
                print(" ")
                if config.local_rank == -1:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # evaluate(args, model, tokenizer)
                    test_acc = get_test_result(test_iter, test_set)
                    print("The test acc is: %.5f" % test_acc)

            if config.local_rank in [-1, 0] and config.save_steps > 0 and global_step % config.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(config.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)

                # torch.save(args, os.path.join(output_dir, "training_args.bin"))

                logger.info("Saving model checkpoint to %s", output_dir)
                # tokenizer.save_vocabulary(output_dir)
                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                # logger.info("Saving optimizer and scheduler states to %s", output_dir)

