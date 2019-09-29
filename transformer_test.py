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
        target_vocab_size=2 ** 13, max_subword_length=1)  # 有需要可以調整字典大小, 每一个中文字是一个单位

    # 將字典檔案存下以方便下次 warmstart
    subword_encoder_zh.save_to_file(zh_vocab_file)

print(f"中文字典大小：{subword_encoder_zh.vocab_size}")
# print(f"前 10 個 subwords：{subword_encoder_en.subwords[:10]}")
# print()
end = time.process_time()
print("耗时：", end - start)

string = sample_examples[0]
zh_string = string[1]
print("each in sample_example:", string, 10 * '-', "\nthe Chinese part: ", zh_string, 10 * '-',
      "\nis the item in sample_example a tuple?", isinstance(string, tuple))
sample_string = sample_examples[0][1]
indices = subword_encoder_zh.encode(sample_string)
print("index of the string: ", indices)

for index in indices:
    print(index, 5 * ' ', subword_encoder_zh.decode([index]))

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
print(100 * '-')


# pre-process:
# insert a special token in both the beginning and the end of seq:
def encode(en_t, zh_t):  # now the en_t,zh_t are eager tensor
    # 因為字典的索引從 0 開始，
    # 我們可以使用 subword_encoder_en.vocab_size 這個值作為 BOS 的索引值
    # 用 subword_encoder_en.vocab_size + 1 作為 EOS 的索引值
    en_indices = [subword_encoder_en.vocab_size] + subword_encoder_en.encode(
        en_t.numpy()) + [subword_encoder_en.vocab_size + 1]
    # 同理，不過是使用中文字典的最後一個索引 + 1
    zh_indices = [subword_encoder_zh.vocab_size] + subword_encoder_zh.encode(
        zh_t.numpy()) + [subword_encoder_zh.vocab_size + 1]

    return en_indices, zh_indices


en, zh = next(iter(
    train_examples))  # here en,zh are just Tensor:<tf.Tensor: id=248, shape=(), dtype=string, numpy=b'Making Do With More'>
en_t, zh_t = encode(en, zh)
pprint((en, zh))
print("after pre-process:")
pprint((en_t, zh_t))


def tf_encode(en_t,
              zh_t):  # because in the dataset.map(), which is run in Graph mode instead of eager mode, so the en_t, zh_t are not eager tensor, which do not contain the .numpy()
    return tf.py_function(encode, [en_t, zh_t], [tf.int64,
                                                 tf.int64])  # this will wrap the encode() into a eager mode enabled function in Graph mode when do the map() later on.


train_dataset = train_examples.map(tf_encode)
print(100 * '-')
print("after pre-processed the whole trainning dataset: (take one pair example)")
en_indices, zh_indices = next(iter(train_dataset))
pprint((en_indices.numpy(), zh_indices.numpy()))
print(100*'-')


# filter the sequences that more than 40 tokens.
print("filter the sequences that more than 40: ")
MAX_LENGTH = 40

def filter_max_length(en, zh, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(en) <= max_length,
                          tf.size(zh) <= max_length)


train_dataset = train_dataset.filter(filter_max_length)

# check after the filter
num_of_data = 0
num_of_invaild = 0
for each in train_dataset:
    en,zh = each
    if tf.size(en) <= MAX_LENGTH and tf.size(zh) <= MAX_LENGTH:
        num_of_data+=1
    else:
        num_of_invaild+=1

print(f"the train_dateset has {num_of_invaild} invalid data, and total {num_of_data} remained valid data")
print(100*'-')

# construct a batch
# when constructing a batch, the length of each sequence need to be padded, so that there are in the same shape
print("after batch and pad: ")
BATCH_SIZE = 64
train_dataset = train_dataset.padded_batch(BATCH_SIZE,padded_shapes=([-1],[-1]))
en_batch,zh_batch = next(iter(train_dataset))
print("English batch:")
print(en_batch)
print(20*'-')
print("Chinese batch:")
print(zh_batch)
