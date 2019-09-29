import tensorflow as tf
import tensorflow_datasets as tfds
import load_dictionary as ld
import load_dataset as lds
import logging

logging.basicConfig(level=logging.ERROR)
MAX_LENGTH = 40
BATCH_SIZE = 128
BUFFER_SIZE = 15000

en_dict, zh_dict = ld.load_dictionary()
train_dataset, val_dataset, _ = lds.load_dataset()
print(train_dataset)


def encode(en_t, zh_t):  # now the en_t,zh_t are eager tensor
    # 因為字典的索引從 0 開始，
    # 我們可以使用 subword_encoder_en.vocab_size 這個值作為 BOS 的索引值
    # 用 subword_encoder_en.vocab_size + 1 作為 EOS 的索引值
    en_indices = [en_dict.vocab_size] + en_dict.encode(
        en_t.numpy()) + [en_dict.vocab_size + 1]
    # 同理，不過是使用中文字典的最後一個索引 + 1
    zh_indices = [zh_dict.vocab_size] + zh_dict.encode(
        zh_t.numpy()) + [zh_dict.vocab_size + 1]

    return en_indices, zh_indices


def tf_encode(en_t,
              zh_t):  # because in the dataset.map(), which is run in Graph mode instead of eager mode, so the en_t, zh_t are not eager tensor, which do not contain the .numpy()
    return tf.py_function(encode, [en_t, zh_t], [tf.int64,
                                                 tf.int64])  # this will wrap the encode() into a eager mode enabled function in Graph mode when do the map() later on.


def filter_max_length(en, zh, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(en) <= max_length,
                          tf.size(zh) <= max_length)


# 訓練集
train_dataset = (train_dataset  # 輸出：(英文句子, 中文句子)
                 .map(tf_encode) # 輸出：(英文索引序列, 中文索引序列)
                 .filter(filter_max_length) # 同上，且序列長度都不超過 40
                 .cache() # 加快讀取數據
                 .shuffle(BUFFER_SIZE) # 將例子洗牌確保隨機性
                 .padded_batch(BATCH_SIZE, # 將 batch 裡的序列都 pad 到一樣長度
                               padded_shapes=([-1], [-1]))
                 .prefetch(tf.data.experimental.AUTOTUNE)) # 加速
# 驗證集
val_dataset = (val_dataset
               .map(tf_encode)
               .filter(filter_max_length)
               .padded_batch(BATCH_SIZE,
                             padded_shapes=([-1], [-1])))

print(train_dataset)
en_batch, zh_batch = next(iter(train_dataset))
print("英文索引序列的 batch")
print(en_batch)
print('-' * 20)
print("中文索引序列的 batch")
print(zh_batch)