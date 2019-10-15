import tensorflow as tf
import tensorflow_datasets as tfds
import load_dictionary as ld
import load_dataset as lds
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from CustomSchedule import CustomSchedule
from MultiHeadAttention import MultiHeadAttention
from loss_function import loss_function
from loss_object import loss_object
from scaled_dot_product_attention import scaled_dot_product_attention
from split_heads import split_heads
from pprint import pprint
from position_wise_feed_forward_network import point_wise_feed_forward_network
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
from Encoder import Encoder
from Decoder import Decoder
from Transformer import Transformer

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
demo_examples = demo_examples.padded_batch(batch_size, padded_shapes=([-1], [-1]))

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
print("mask:", mask)
print("wq.shape:", wq.shape)
print("wk.shape:", wk.shape)
print("wv.shape:", wv.shape)
print("output.shape:", output.shape)
print("scaled output.shape:", scaled_attention.shape)
print("attention_weights.shape:", attention_weights.shape)
print("scaled_attention_logits.shape:", scaled_attention_logits.shape)

print("\noutput:", output)
print(100 * '-')

# test the position_wise feed forward network
print("test the position_wise_feed_forward_network:")
batch_size = 64
seq_len = 10
d_model = 512
dff = 2048

x = tf.random.uniform((batch_size, seq_len, d_model))
ffn = point_wise_feed_forward_network(d_model, dff)
output = ffn(x)
print("input shape:", x.shape, "\n")
print("input:", x)
print("\noutput shape:", output.shape)
print("\noutput:", output)
print(100 * '-')

# encoder layer: 一個 Encoder layer 裡頭會有兩個 sub-layers，分別為 MHA 以及 FFN。在 Add & Norm 步驟裡頭，每個 sub-layer 會有一個殘差連結（residual
# connection）來幫助減緩梯度消失（Gradient Vanishing）的問題。接著兩個 sub-layers 都會針對最後一維 d_model 做 layer normalization，將 batch
# 裡頭每個子詞的輸出獨立做轉換，使其平均與標準差分別靠近 0 和 1 之後輸出。
print("EncoderLayer: \n")
# hyper-parameters:
d_model = 4
num_heads = 2
dff = 8

# construct a encoder layer that uses the above inputs:
enc_layer = EncoderLayer(d_model, num_heads, dff)
en_padding_mask = create_padding_mask(en)
enc_out = enc_layer(emb_en, training=False, mask=en_padding_mask)

print("en:", en, "\n")
print(20 * '-')
print("en_padding_mask:", padding_mask, "\n")
print(20 * '-')
print("emb_en:", emb_en, "\n")
print(20 * '-')
print("enc_out:", enc_out, "\n")
print(emb_en.shape == enc_out.shape)
print(100 * '-')

# combined_mask in DecoderLayer:
print("Combined_mask in DecoderLayer:")
"""
combined mask is the maximum of the two masks: look_ahead_mask and padding_mask
"""
zh_padding_mask = create_padding_mask(zh)
look_ahead_mask = create_look_ahead_mask(zh.shape[-1])
combined_mask = tf.maximum(zh_padding_mask, look_ahead_mask)

print("zh:", zh, "\n")
print(20 * '-')
print("zh_padding_mask:", zh_padding_mask, "\nzh_padding_mask.shape:", zh_padding_mask.shape)
print(20 * '-')
print("look_ahead_mask:", look_ahead_mask, "\n")
print(20 * '-')
print("combined_mask:", combined_mask, "\n")
print(100 * '-')

print("DecoderLayer: \n")
# hyperparameters:
d_model = 4
num_heads = 2
dff = 8

# construt decoder layer
dec_layer = DecoderLayer(d_model, num_heads, dff)

# create masks
zh_padding_mask = create_padding_mask(zh)
look_ahead_mask = create_look_ahead_mask(zh.shape[-1])
combined_mask = tf.maximum(zh_padding_mask, look_ahead_mask)

# init decoder layer
dec_out, dec_self_attention_weights, dec_enc_attention_weights = dec_layer(emb_zh, enc_out, False, combined_mask,
                                                                           en_padding_mask)

print("emb_zh:", emb_zh)
print(20 * '-')
print("enc_out:", enc_out)
print(20 * '-')
print("dec_out:", dec_out)
print(enc_out.shape == dec_out.shape)
print(20 * '-')
print("dec_self_attention_weights.shape:", dec_self_attention_weights.shape)
print(20 * '-')
print("dec_enc_attention_weights.shape:", dec_enc_attention_weights.shape)
print(100 * '-')

# test Encoder
print("Encoder:\n")
# 超參數
num_layers = 2  # 2 層的 Encoder
d_model = 4
num_heads = 2
dff = 8
input_vocab_size = vocab_size_en  # 記得加上 <start>, <end>

# 初始化一個 Encoder
encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size)

# 將 2 維的索引序列丟入 Encoder 做編碼
enc_out = encoder(en, training=False, mask=en_padding_mask)
print("en:", en)
print("-" * 20)
print("enc_out:", enc_out)
print(100 * '-')

# test decoder
print("Decoder:\n")
# 超參數
num_layers = 2  # 2 層的 Decoder
d_model = 4
num_heads = 2
dff = 8
target_vocab_size = vocab_size_zh  # 記得加上 <start>, <end>

# 遮罩
en_padding_mask = create_padding_mask(en)
zh_padding_mask = create_padding_mask(zh)
look_ahead_mask = create_look_ahead_mask(zh.shape[1])
combined_mask = tf.math.maximum(zh_padding_mask, look_ahead_mask)

# 初始化一個 Decoder
decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size)

# 將 2 維的索引序列以及遮罩丟入 Decoder
print("zh:", zh)
print("-" * 20)
print("combined_mask:", combined_mask)
print("-" * 20)
print("enc_out:", enc_out)
print("-" * 20)
print("en_padding_mask:", en_padding_mask)
print("-" * 20)
dec_out, attention_weights = decoder(zh, enc_out, training=False,
                                     combined_mask=combined_mask,
                                     inp_padding_mask=en_padding_mask)
print("dec_out:", dec_out)
print("-" * 20)
for block_name, attn_weights in attention_weights.items():
    print(f"{block_name}.shape: {attn_weights.shape}")
print(100 * '-')

# 测试transformer
print("test with the transformer:\n")
# 超參數
num_layers = 1
d_model = 4
num_heads = 2
dff = 8

tar_train = zh[:, :-1]  # means in the second dim, right-shift one pos (the beginning of LHS minus 1)
tar_real = zh[:, 1:]  # means in the second dim, left-shift one pos (the beginning of RHS plus 1)
# try to read one char in the tar_train, then generate the next char, so the next
# char it generate should be compared with the sequence after left-shift, that is
# tar_real.

inp_padding_mask = create_padding_mask(en)
tar_padding_mask = create_padding_mask(tar_train)
look_ahead_mask = create_look_ahead_mask(tar_train.shape[1])
combined_mask = tf.math.maximum(tar_padding_mask, look_ahead_mask)

transformer = Transformer(num_layers, d_model, num_heads, dff, vocab_size_en, vocab_size_zh)

predictions, attention_weights = transformer(en, tar_train, False, inp_padding_mask, combined_mask, inp_padding_mask)

print("tar:", zh)
print(20 * '-')
print("tar_train:", tar_train)
print(20 * '-')
print("tar_real:", tar_real)
print(20 * '-')
print("prediction:", predictions)
print(100 * '-')

# loss_cross_entropy
print("sparse cross entropy:")
real = tf.constant([2, 1, 2], dtype=tf.float32)
# represent the expected pos of the largest number in sequence.
pred = tf.constant([[3, 0, 5], [0, 1, 0], [0, 1, 3]], dtype=tf.float32)
# this will be computed by softmax,
# then come out the pos of largest one, for
# example [3, 0, 5], the largest is 5 and its index is 2, so the cost of this is small because the expected pos is
# also 2
print(loss_object(real, pred))  # tf.Tensor([2.1328452  0.55144465 3.169846  ], shape=(3,), dtype=float32)
print(loss_function(real,pred))
print(100*'-')

# CustomSchedule
# 這 schedule 讓訓練過程的前 warmup_steps 的 learning rate 線性增加，
# 在那之後則跟步驟數 step_num 的反平方根成比例下降。
print("CustomSchedule:")
d_models = [128, 256, 512]
warmup_steps = [1000 * i for i in range(1, 4)]

schedules = []
labels = []
colors = ["blue", "red", "black"]
for d in d_models:
  schedules += [CustomSchedule(d, s) for s in warmup_steps]
  labels += [f"d_model: {d}, warm: {s}" for s in warmup_steps]

for i, (schedule, label) in enumerate(zip(schedules, labels)):
  plt.plot(schedule(tf.range(10000, dtype=tf.float32)),
           label=label, color=colors[i // 3])

plt.legend()

plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()
# 將客製化 learning rate schdeule 丟入 Adam opt.
# Adam opt. 的參數都跟論文相同
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)
print(100*'-')

# Actual trainning for Transformer
print("Actural trainning:")
print(20*'-')
print("Hyper-Parameters:")
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = vocab_size_en
target_vocab_size = vocab_size_zh
dropout_rate = 0.1  # default value

# 論文裡頭最基本的 Transformer 配置為：
#
# num_layers=6
# d_model=512
# dff=2048

print(f"This transformer has {num_layers} of Encoder/Decoder layers:",
       "\nd_model:",d_model,
       "\nnum_heads",num_heads,
       "\ndff:",dff,
       "\nnum_heads:",num_heads,
       "\ninput_vocab_size:",input_vocab_size,
       "\ntarget_vocab_size:",target_vocab_size,
       "\ndropout:",dropout_rate,
       "\n")

print(20*'-')
transformer = Transformer(num_layers,d_model,num_heads,dff,input_vocab_size,target_vocab_size,dropout_rate)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

"""
設置 checkpoint 來定期儲存 / 讀取模型及 optimizer 是必備的。
我們在底下會定義一個 checkpoint 路徑，此路徑包含了各種超參數的資訊，
方便之後比較不同實驗的結果並載入已訓練的進度。
我們也需要一個 checkpoint manager 來做所有跟存讀模型有關的雜事，
並只保留最新 5 個 checkpoints 以避免佔用太多空間：
"""
# 方便比較不同實驗/ 不同超參數設定的結果
output_dir = "nmt"
checkpoint_path = os.path.join(output_dir, "checkpoints")
log_dir = os.path.join(output_dir, 'logs')
train_perc = 20
run_id = f"{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff_{train_perc}train_perc"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)

# tf.train.Checkpoint 可以幫我們把想要存下來的東西整合起來，方便儲存與讀取
# 一般來說你會想存下模型以及 optimizer 的狀態
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

# ckpt_manager 會去 checkpoint_path 看有沒有符合 ckpt 裡頭定義的東西
# 存檔的時候只保留最近 5 次 checkpoints，其他自動刪除
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果在 checkpoint 路徑上有發現檔案就讀進來
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

    # 用來確認之前訓練多少 epochs 了
    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print(f'已讀取最新的 checkpoint，模型已訓練 {last_epoch} epochs。')
else:
    last_epoch = 0
    print("沒找到 checkpoint，從頭訓練。")

# create masks
# 為 Transformer 的 Encoder / Decoder 準備遮罩
def create_masks(inp, tar):
    # 英文句子的 padding mask，要交給 Encoder layer 自注意力機制用的
    enc_padding_mask = create_padding_mask(inp)

    # 同樣也是英文句子的 padding mask，但是是要交給 Decoder layer 的 MHA 2
    # 關注 Encoder 輸出序列用的
    dec_padding_mask = create_padding_mask(inp)

    # Decoder layer 的 MHA1 在做自注意力機制用的
    # `combined_mask` 是中文句子的 padding mask 跟 look ahead mask 的疊加
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

# define train_step:
"""
一個數據集包含多個 batch，而每次拿一個 batch 來訓練的步驟就稱作 train_step。
為了讓程式碼更簡潔以及容易優化，我們會定義 Transformer 在一次訓練步驟（處理一個 batch）所需要做的所有事情。
不限於 Transformer，一般來說 train_step 函式裡會有幾個重要步驟：
1.對訓練數據做些必要的前處理
2.將數據丟入模型，取得預測結果
3.用預測結果跟正確解答計算 loss
4.取出梯度並利用 optimizer 做梯度下降
"""
# define tensorboard
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
# -------------------
"""
train_step 函式的寫法非常固定：
1.對輸入數據做些前處理（本文中的遮罩、將輸出序列左移當成正解 etc.）
2.利用 tf.GradientTape 輕鬆記錄數據被模型做的所有轉換並計算 loss
3.將梯度取出並讓 optimzier 對可被訓練的權重做梯度下降（上升）
"""

@tf.function  # 讓 TensorFlow 幫我們將 eager code 優化並加快運算
def train_step(inp, tar):
    # 前面說過的，用去尾的原始序列去預測下一個字的序列
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    # 建立 3 個遮罩
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # 紀錄 Transformer 的所有運算過程以方便之後做梯度下降
    with tf.GradientTape() as tape:
        # 注意是丟入 `tar_inp` 而非 `tar`。記得將 `training` 參數設定為 True
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        # 跟影片中顯示的相同，計算左移一個字的序列跟模型預測分佈之間的差異，當作 loss
        loss = loss_function(tar_real, predictions)  # use sparse_categories_cross_entropy

    # 取出梯度並呼叫前面定義的 Adam optimizer 幫我們更新 Transformer 裡頭可訓練的參數
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 將 loss 以及訓練 acc 記錄到 TensorBoard 上，非必要
    train_loss(loss)
    train_accuracy(tar_real, predictions)

# define epochs
"""
這邊的邏輯也很簡單，在每個 epoch 都：

1.（非必要）重置寫到 TensorBoard 的 metrics 的值
2. 將整個數據集的 batch 取出，交給 train_step 函式處理
3.（非必要）存 checkpoints
4.（非必要）將當前 epoch 結果寫到 TensorBoard
5.（非必要）在標準輸出顯示當前 epoch 結果

simple version:
for epoch in range(EPOCHS):
  for inp, tar in train_dataset:
    train_step(inp, tar)
"""
EPOCHS = 30
print(f"this hyper-parameter based Transformer has already trained for {last_epoch} epochs.")
print(f"the last epochs: {min(0,last_epoch - EPOCHS)}")

# 用來寫資訊到 TensorBoard，非必要但十分推薦
summary_writer = tf.summary.create_file_writer(log_dir)

# 比對設定的 `EPOCHS` 以及已訓練的 `last_epoch` 來決定還要訓練多少 epochs
for epoch in range(last_epoch, EPOCHS):
    start = time.time()

    # 重置紀錄 TensorBoard 的 metrics
    train_loss.reset_states()
    train_accuracy.reset_states()

    # 一個 epoch 就是把我們定義的訓練資料集一個一個 batch 拿出來處理，直到看完整個數據集
    for (step_idx, (inp, tar)) in enumerate(train_dataset):
        # 每次 step 就是將數據丟入 Transformer，讓它生預測結果並計算梯度最小化 loss
        train_step(inp, tar)

        # 每個 epoch 完成就存一次檔
    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    # 將 loss 以及 accuracy 寫到 TensorBoard 上
    with summary_writer.as_default():
        tf.summary.scalar("train_loss", train_loss.result(), step=epoch + 1)
        tf.summary.scalar("train_acc", train_accuracy.result(), step=epoch + 1)

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))





