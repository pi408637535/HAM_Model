#coding:utf-8
from torch.utils import data
import os
import torch
import nltk
import numpy as np
from transformers import BertTokenizer
import codecs
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

PAD,CLS,SEP = "[PAD]","[CLS]","[SEP]"

class IMDB_Data(data.DataLoader):
    def __init__(self,data_name,min_count, tokenizer,word2id = None,max_sentence_length = 100, batch_size=64):
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        self.data_name = "/toutiao/"+data_name

        self.label_dic = [x.strip() for x in open( self.path +
             '/toutiao/class.txt').readlines()]
        self.label_dic = { k:v for v,k in enumerate(self.label_dic) }

        self.min_count = min_count
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.datas, self.segment_ids,self.input_masks,self.labels = self.load_data()

        for i in range(len(self.datas)):
            self.datas[i] = np.array(self.datas[i])
            self.segment_ids[i] = np.array(self.segment_ids[i])
            self.input_masks[i] = np.array(self.input_masks[i])
            # self.datas[i] = torch.Tensor(self.datas[i])
            # self.segment_ids[i] = torch.Tensor(self.segment_ids[i])
            # self.input_masks[i] = torch.Tensor(self.input_masks[i])



    def load_data(self):

        datas = []
        labels = []
        with codecs.open(self.path+self.data_name, "r", encoding="utf-8") as fin:
            for ele in fin:
                if len(ele) < 10:
                    continue
                data,label = ele.split("\t")
                data = data.split()
                if not data:
                    continue
                datas.append(data[0])
                labels.append(self.label_dic[label[:-1]])

        datas = [data.split("。") for data in datas]
        datas = sorted(datas,key = lambda x:len(x),reverse=True)

        # for i,data in enumerate(datas):
        #     for j,sentence in enumerate(datas[i]):
        #
        #         datas[i][j] = sentence

        datas, segment_ids, input_masks = self.convert_data2id(datas)
        return datas,segment_ids,input_masks,labels


    def convert_data2id(self, datas, sequence_a_segment_id=0, pad_token=0,
                        pad_token_segment_id=0,mask_padding_with_zero=True):
        # input_ids, input_mask, segment_ids
        input_masks = [ [] for i in range(len(datas)) ]
        segment_ids =  [ [] for i in range(len(datas)) ]
        for i,document in enumerate(datas):
            if i%10000==0:
                print (i,len(datas))
            for j,sentence in enumerate(document):

                special_tokens_count = 2
                if len(sentence) > self.max_sentence_length - special_tokens_count:
                    sentence = sentence[: (self.max_sentence_length - special_tokens_count)]

                sentence = self.tokenizer.tokenize(sentence)
                sentence += [SEP]
                sentence = [CLS] + sentence
                segment_id = [sequence_a_segment_id] * len(sentence)

                input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                padding_length = self.max_sentence_length - len(input_ids)

                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_id += [pad_token_segment_id] * padding_length



                assert len(input_ids) == self.max_sentence_length
                assert len(input_mask) == self.max_sentence_length
                assert len(segment_id) == self.max_sentence_length

                datas[i][j] = input_ids
                #存入一个数组，维度扩充一
                input_masks[i].append(input_mask)
                segment_ids[i].append(segment_id)

        # 对同一batch下数据进行 paragraph 填充
        for i in range(0, len(datas), self.batch_size):
            max_data_length = max([len(x) for x in datas[i:i+self.batch_size]])
            for j in range(i, min(i+self.batch_size,len(datas))):
                # [ [PAD] * sentence ] * pad_paragraph
                datas[j] = datas[j] + [ [self.tokenizer.convert_tokens_to_ids(PAD)]*self.max_sentence_length]* (max_data_length-len(datas[j]))
                datas[j] = datas[j]

                input_masks[j] = input_masks[j] + [ [0 if mask_padding_with_zero else 1] * self.max_sentence_length] *(max_data_length-len(input_masks[j]))
                input_masks[j] = input_masks[j]

                segment_ids[j] = segment_ids[j] + [ [pad_token_segment_id] * self.max_sentence_length] * (max_data_length - len(segment_ids[j]))
                segment_ids[j] = segment_ids[j]
                segment_ids



        return datas,segment_ids,input_masks


    def __getitem__(self, idx):
        return self.datas[idx],self.segment_ids[idx],self.input_masks[idx],self.labels[idx]
        # return self.datas[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# def collate_fn(batch):
#     """
#     batch should be a list of (sequence, target, length) tuples...
#     Returns a padded tensor of sequences sorted from longest to shortest,
#     """
#     all_input_ids, all_attention_mask, all_token_type_ids, all_labels = map(torch.stack, zip(*batch))
#     max_len = max(all_input_ids).item()
#     all_input_ids = all_input_ids[:, :max_len]
#     all_attention_mask = all_attention_mask[:, :max_len]
#     all_token_type_ids = all_token_type_ids[:, :max_len]
#     all_labels = all_labels[:,:max_len]
#     return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


if __name__=="__main__":
    model_name_or_path = os.path.abspath("..") + "/prev_trained_model/bert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    imdb_data = IMDB_Data(data_name="toutiao_article_score_train.txt", tokenizer=tokenizer, min_count=5)
    training_iter = torch.utils.data.DataLoader(dataset=imdb_data,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=0)
    for data,maks,segment,label in training_iter:
        print (np.array(data).shape, np.array(maks).shape, np.array(segment).shape,)
