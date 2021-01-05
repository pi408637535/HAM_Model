# —*- coding: utf-8 -*-

import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=10, help="embedding size of word embedding")
    parser.add_argument("--epoch",type=int,default=200,help="epoch of training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=2,help="gpu num")
    parser.add_argument("--batch_size",type=int,default=2,help="batch size during training")
    parser.add_argument("--seed",type=int,default=0,help="seed of random")
    parser.add_argument("--min_count",type=int,default=5,help="min count of words")
    parser.add_argument("--max_sentence_length",type=int,default=100,help="max sentence length")
    parser.add_argument("--embedding_size",type=int,default=200,help="word embedding size")
    parser.add_argument("--gru_size",type=int,default=50,help="gru size")
    parser.add_argument("--class_num",type=int,default=8,help="class num")

    parser.add_argument("--weight_decay", default=0.01, type=float,help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--score_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")

    return parser.parse_args()
