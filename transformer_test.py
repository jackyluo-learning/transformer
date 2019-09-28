import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from pprint import pprint
from IPython.display import clear_output

print(tf.__version__)
logging.basicConfig(level="ERROR")  # change the log level to error
np.set_printoptions(suppress=True)  # let the numpy do not print out as the scientific pattern

# setting the file location and variables
output_dir = "nmt"  # where the generated dictionary will locate
en_vocab_file = os.path.join(output_dir, "en_vocab")  # join the two sub-directories
zh_vocab_file = os.path.join(output_dir, "zh_vocab")
checkpoint_path = os.path.join(output_dir, "checkpoints")
log_dir = os.path.join(output_dir, 'logs')
download_dir = "tensorflow-datasets/downloads"  # where the dataset will locate in the project.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tmp_builder = tfds.builder("wmt19_translate/zh-en")  # check what are the dataset inside the WMT2019
# pprint(tmp_builder.subsets)

# download the dataset
config = tfds.translate.wmt.WmtConfig(
    version=tfds.core.Version('0.0.3', experiments={tfds.core.Experiment.S3: False}),
    language_pair=("zh", "en"),
    subsets={
        tfds.Split.TRAIN: ["newscommentary_v14"]  # select the news comment train dataset to be the dataset,
    }
)
builder = tfds.builder("wmt_translate", config=config)  # fetch the dataset
builder.download_and_prepare(download_dir=download_dir)  # download it, and write it to disk
clear_output()

# cut the dataset into three parts: train, validate, and the last one we drop it away
train_perc = 20
val_prec = 1
drop_prec = 100 - train_perc - val_prec

split = tfds.Split.TRAIN.subsplit([train_perc, val_prec, drop_prec])  # define a split that cut the dataset
split

examples = builder.as_dataset(split=split, as_supervised=True)  # use the split above as a parameter
train_examples, val_examples, _ = examples
# print(train_examples)
# print(val_examples)

# output some data as example
sample_examples = []
for en, zh in train_examples.take(3):
    en = en.numpy().decode("utf-8")  # use numpy().decode the string into utf-8 format
    zh = zh.numpy().decode("utf-8")

    # print(en)
    # print(zh)
    # print('-' * 10)

    sample_examples.append((en, zh))  # append a tuple into a list

# construct the dictionary of Chinese and English

start = time.process_time()
try:
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_vocab_file)
    print(f"載入已建立的英文字典： {en_vocab_file}")
except:
    print("沒有已建立的英文字典，從頭建立。")
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for en, _ in train_examples),
        target_vocab_size=2 ** 13)  # 有需要可以調整字典大小

    # 將字典檔案存下以方便下次 warmstart
    subword_encoder_en.save_to_file(en_vocab_file)

print(f"英文字典大小：{subword_encoder_en.vocab_size}")
# print(f"前 10 個 subwords：{subword_encoder_en.subwords[:10]}")
# print()
end = time.process_time()
print(end - start)

# test with a sentence
sample_string = 'Guangzhou is beautiful.'
indices = subword_encoder_en.encode(sample_string)
print("sample string: ", sample_string, "index of sample string: ", indices)
print(100 * '-')

# recover from the indices
for index in indices:
    print(index, 5 * ' ', subword_encoder_en.decode([index]))

# construct a Chinese dictionary
start = time.process_time()
try:
  subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
  print(f"載入已建立的中文字典： {zh_vocab_file}")
except:
  print("沒有已建立的中文字典，從頭建立。")
  subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (zh.numpy() for _, zh in train_examples),  # here should be _, zh, as the pair in training_set is like en-zh
    target_vocab_size=2 ** 13,max_subword_length=1)  # 有需要可以調整字典大小, 每一个中文字是一个单位

  # 將字典檔案存下以方便下次 warmstart
  subword_encoder_zh.save_to_file(zh_vocab_file)

print(f"中文字典大小：{subword_encoder_zh.vocab_size}")
# print(f"前 10 個 subwords：{subword_encoder_en.subwords[:10]}")
# print()
end = time.process_time()
print("耗时：", end-start)

string = sample_examples[0]
zh_string = string[1]
print("each in sample_example:",string,10*'-',"\nthe Chinese part: ",zh_string,10*'-', "\nis the item in sample_example a tuple?", isinstance(string,tuple))
sample_string = sample_examples[0][1]
indices = subword_encoder_zh.encode(sample_string)
print("index of the string: ",indices)

for index in indices:
    print(index, 5*' ', subword_encoder_zh.decode([index]))

en = "The eurozone’s collapse forces a major realignment of European politics."
zh = "欧元区的瓦解强迫欧洲政治进行一次重大改组。"

# 將文字轉成為 subword indices
en_indices = subword_encoder_en.encode(en)
zh_indices = subword_encoder_zh.encode(zh)

print("[英中原文]（轉換前）")
print(en)
print(zh)
print()
print('-' * 20)
print()
print("[英中序列]（轉換後）")
print(en_indices)
print(zh_indices)

# pre-process:
# insert a special token in both the beginning and the end of seq:

