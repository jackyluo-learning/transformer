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

# setting the download file location and variables
output_dir = "nmt"
en_vocab_file = os.path.join(output_dir, "en_vocab")  # join the two sub-directories
zh_vocab_file = os.path.join(output_dir, "zh_vocab")
checkpoint_path = os.path.join(output_dir, "checkpoints")
log_dir = os.path.join(output_dir, 'logs')
download_dir = "tensorflow-datasets/downloads"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tmp_builder = tfds.builder("wmt19_translate/zh-en")  # check what are the dataset inside the WMT2019
pprint(tmp_builder.subsets)

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
print(train_examples)
print(val_examples)

# output some data as example
sample_examples = []
for en, zh in train_examples.take(3):
    en = en.numpy().decode("utf-8")  # use numpy().decode the string into utf-8 format
    zh = zh.numpy().decode("utf-8")

    print(en)
    print(zh)
    print('-' * 10)

    sample_examples.append((en, zh))

# construct the dictionary of Chinese and English

# start = time.process_time()
try:
  subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_vocab_file)
  print(f"載入已建立的字典： {en_vocab_file}")
except:
  print("沒有已建立的字典，從頭建立。")
  subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for en, _ in train_examples),
    target_vocab_size=2 ** 13)  # 有需要可以調整字典大小

  # 將字典檔案存下以方便下次 warmstart
  subword_encoder_en.save_to_file(en_vocab_file)

print(f"字典大小：{subword_encoder_en.vocab_size}")
print(f"前 10 個 subwords：{subword_encoder_en.subwords[:10]}")
print()
# end = time.process_time()
# print(end-start)