import tensorflow as tf
import tensorflow_datasets as tfds
import load_dictionary as ld
import load_dataset as lds
import logging
import numpy as np
from pprint import pprint

logging.basicConfig(level=logging.ERROR)
np.set_printoptions(suppress=True)

demo_examples = [
    ("It is important.", "这很重要。"),
    ("The numbers speak for themselves.", "数字证明了一切。"),
]
pprint(demo_examples)

batch_size = 2
# demo_examples = tfds.from_tensor_slices()
demo_examples = tf.data.Dataset.from_tensor_slices((
    [en for en, _ in demo_examples], [zh for _, zh in demo_examples]
))
print(demo_examples)

en_dict, zh_dict = ld.load_dictionary()
train_dataset, val_dataset, _ = lds.load_dataset()


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


demo_examples = demo_examples.map(tf_encode)
demo_examples = demo_examples.padded_batch(batch_size, padded_shapes=([10], [10]))

en, zh = next(iter(demo_examples))
print("encode the two set of en-zh sentence: ")
pprint((en, zh))
print(100 * '-')

# embedding:
vocab_size_en = en_dict.vocab_size + 2  # because add two more tokens: <start>,<end>
vocab_size_zh = zh_dict.vocab_size + 2

# transform each word in dictionary from one dim to 4 dim,
# by create a embedding layer
d_model = 4
embedding_layer_en = tf.keras.layers.Embedding(vocab_size_en, d_model)
embedding_layer_zh = tf.keras.layers.Embedding(vocab_size_zh, d_model)

emb_en = embedding_layer_en(en)
emb_zh = embedding_layer_zh(zh)
print("after embedding:")
print(emb_en)
print(emb_zh)
print(100 * '-')


# padding mask: mask those pad with 0
def create_padding_mask(seq):
    # padding mask 的工作就是把索引序列中為 0 的位置設為 1
    mask = tf.cast(tf.equal(seq, 0), tf.float32)  # tf.equal(): compare each in seq to 0, equal return True.
    return mask[:, tf.newaxis, tf.newaxis, :]  # broadcasting


en_mask = create_padding_mask(en)
print(en_mask)

print("en:", en)
print("-" * 100)
print("tf.squeeze(en_mask):", tf.squeeze(en_mask))