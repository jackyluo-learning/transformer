import tensorflow as tf
import tensorflow_datasets as tfds
import load_dictionary as ld
import load_dataset as lds
import logging
import numpy as np

from MultiHeadAttention import MultiHeadAttention
from scaled_dot_product_attention import scaled_dot_product_attention
from split_heads import split_heads
from pprint import pprint
from position_wise_feed_forward_network import point_wise_feed_forward_network

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
print(100 * '-')

# 注意力機制（或稱注意函式，attention function）概念上就是拿一個查詢（query）去跟一組 key-values 做運算，最後產生一個輸出。只是我們會利用矩陣運算同時讓多個查詢跟一組 key-values
# 做運算，最大化計算效率。 Scaled dot product attention 跟以往 multiplicative attention 一樣是先將維度相同的 Q 跟 K
# 做點積：將對應維度的值兩兩相乘後相加得到單一數值，接著把這些數值除以一個 scaling factor sqrt(dk) ，然後再丟入 softmax 函式得到相加為 1 的注意權重（attention weights）。
# scaled dot product attention
tf.random.set_seed(9527)  # set a seed tha can enable us to get the same value every time.
q = emb_en
k = emb_en
# generate a tensor that has the same shape of emb_en
v = tf.cast(tf.math.greater(tf.random.uniform(shape=emb_en.shape), 0.5), tf.float32)
print(v)

print("scaled dot product attention:")
# q 跟 k 都代表同個張量 emb_inp，因此 attention_weights 事實上就代表了 emb_inp 裡頭每個英文序列中的子詞對其他位置的子詞的注意權重。
# output 則是句子裡頭每個位置的子詞將 attention_weights 當作權重，從其他位置的子詞對應的資訊 v 裡頭抽取有用訊息後匯總出來的結果。
mask = None
output, attention_weights, _ = scaled_dot_product_attention(q, k, v, mask)
print("output:", output)
print("-" * 100)
print("attention_weights:", attention_weights)
print(100 * '-')
# mask in scaled_dot_product_attention
#  q 跟 k 都是從 emb_inp 來的。emb_inp 代表著英文句子的詞嵌入張量，而裡頭的第一個句子應該是有 <pad> token,
# 因此在注意函式裡頭，我們將遮罩乘上一個接近負無窮大的 -1e9，並把它加到進入 softmax 前的 logits 上面。這樣可以讓這些被加上極大負值的位置變得無關緊要，在經過 softmax 以後的值趨近於 0。
output, attention_weights, _ = scaled_dot_product_attention(q, k, v, tf.squeeze(en_mask, axis=1))
print("attention_weights after masked:", attention_weights)
print("output:", output)
print(100 * '-')


# 在 padding mask 的幫助下，注意函式輸出的新序列 output 裡頭的每個子詞都只從序列 k （也就是序列 q 自己）的前 6 個實際子詞而非 <pad> 來獲得語義資訊

# look ahead mask
# 建立一個 2 維矩陣，維度為 (size, size)，
# 其遮罩為一個右上角的三角形
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


seq_len = emb_zh.shape[1]
look_ahead_mask = create_look_ahead_mask(seq_len)
print("emb_zh:", emb_zh)
print(100 * '-')
print("look ahead mask:", look_ahead_mask)
print(100 * '-')

# simulation of decoder
print("simulation of decoder:")
dec_q = dec_k = emb_zh
dec_v = tf.cast(tf.math.greater(tf.random.uniform(shape=emb_zh.shape), 0.5), tf.float32)
print("v:", dec_v)

output, attention_weights, _ = scaled_dot_product_attention(dec_q, dec_k, dec_v, look_ahead_mask)
print("attention_weights:", attention_weights)
# 就跟一般的 Seq2Seq 模型相同，Transformer 裡頭的 Decoder 在生成輸出序列時也是一次產生一個子詞。因此跟輸入的英文句子不同，中文句子裡頭的每個子詞都是在不同時間點產生的。所以理論上 Decoder
# 在時間點 t - 1 （或者說是位置 t - 1）已經生成的子詞 subword_t_minus_1 在生成的時候是不可能能夠關注到下個時間點 t（位置 t）所生成的子詞 subword_t 的，儘管它們在 Transformer
# 裡頭同時被做矩陣運算。
print("attention_weights of the first word:", attention_weights[:, 0, :])
# 兩個句子的第一個子詞因為自己前面已經沒有其他子詞，所以將全部的注意力 1都放在自己身上。
print("attention_weights of the second word:", attention_weights[:, 1, :])
# 兩個句子的第 2 個子詞因為只能看到序列中的第一個子詞以及自己，因此前兩個位置的注意權重加總即為 1，後面位置的權重皆為 0。
print(100 * '-')

# multi-head attention
# 將 Q、K 以及 V 這三個張量先個別轉換到 d_model 維空間，再將其拆成多個比較低維的 depth 維度 N 次以後，將這些產生的小 q、小 k 以及小 v
# 分別丟入前面的注意函式得到 N 個結果。接著將這 N 個 heads 的結果串接起來，最後通過一個線性轉換就能得到 multi-head attention 的輸出
# transform a d_model vectoer into a num_heads * depth vector: (d_model: the dim of each word: 4)
num_heads = 2
x = emb_en
output = split_heads(emb_en, d_model, num_heads)
print("before multi-head transform:", x)
print("after multi-head transform:", output)
print(100 * '-')
# 3 維詞嵌入張量 emb_en 已經被轉換成一個 4 維張量了，且最後一個維度 shape[-1] = 4 被拆成兩半.
# 觀察 split_heads 的輸入輸出，你會發現序列裡每個子詞原來為 d_model 維的
# reprsentation 被拆成多個相同但較短的 depth 維度。而每個 head 的 2 維矩陣事實上仍然代表原來的序列，只是裡頭子詞的 repr. 維度降低了。

# test MultiHeadAttention:
assert d_model == emb_en.shape[-1] == 4
num_heads = 2

print(f"d_model: {d_model}")
print(f"num_heads: {num_heads}\n")

# 初始化一個 multi-head attention layer
mha = MultiHeadAttention(d_model, num_heads)

# 簡單將 v, k, q 都設置為 `emb_inp`
# 順便看看 padding mask 的作用。
# 別忘記，第一個英文序列的最後兩個 tokens 是 <pad>
v = k = q = emb_en
padding_mask = create_padding_mask(en)
print("q.shape:", q.shape)
print("k.shape:", k.shape)
print("v.shape:", v.shape)
print("padding_mask.shape:", padding_mask.shape)

wq, wk, wv, output, scaled_attention, attention_weights, scaled_attention_logits = mha(v, k, q, mask)
print("wq.shape:", wq.shape)
print("wk.shape:", wk.shape)
print("wv.shape:", wv.shape)
print("output.shape:", output.shape)
print("scaled output.shape:", scaled_attention.shape)
print("attention_weights.shape:", attention_weights.shape)
print("scaled_attention_logits.shape:", scaled_attention_logits.shape)

print("\noutput:", output)
print(100*'-')

# test the position_wise feed forward network
print("test the position_wise_feed_forward_network:")
batch_size = 64
seq_len = 10
d_model = 512
dff = 2048

x = tf.random.uniform((batch_size,seq_len,d_model))
ffn = point_wise_feed_forward_network(d_model,dff)
output = ffn(x)
print("input shape:",x.shape,"\n")
print("input:",x)
print("\noutput shape:",output.shape)
print("\noutput:",output)
print(100*'-')

# encoder layer: 一個 Encoder layer 裡頭會有兩個 sub-layers，分別為 MHA 以及 FFN。在 Add & Norm 步驟裡頭，每個 sub-layer 會有一個殘差連結（residual
# connection）來幫助減緩梯度消失（Gradient Vanishing）的問題。接著兩個 sub-layers 都會針對最後一維 d_model 做 layer normalization，將 batch
# 裡頭每個子詞的輸出獨立做轉換，使其平均與標準差分別靠近 0 和 1 之後輸出。