import os
import time

import tensorflow as tf
import logging

from CustomSchedule import CustomSchedule
from Transformer import Transformer
from create_masks import create_padding_mask, create_look_ahead_mask
from load_dataset import load_dataset
from load_dictionary import load_dictionary
from loss_function import loss_function

logging.basicConfig(level=logging.ERROR)
train_examples, test_examples, _ = load_dataset()
sample_sentences = []
print("train_examples:\n", train_examples)
for en_dict, zh_dict in train_examples.take(3):
    en = en_dict.numpy().decode("utf-8")
    zh = zh_dict.numpy().decode("utf-8")
    print(en)
    print(zh)
    sample_sentences.append((en, zh))

print("...Successfully loading dataset...")
print(100 * '-')

en_dict, zh_dict = load_dictionary()
print("English dictionary size:", en_dict.vocab_size)
print("Chinese dictionary size:", zh_dict.vocab_size)
print("...Successfully loading dictionary...")
print(100 * '-')

sample_sentence_en = sample_sentences[0][0]
sample_sentence_zh = sample_sentences[0][1]
print(sample_sentence_en)
print(sample_sentence_zh)
sample_sentence_en_enc = en_dict.encode(sample_sentence_en)
sample_sentence_zh_enc = zh_dict.encode(sample_sentence_zh)
print(sample_sentence_en_enc)
print(sample_sentence_zh_enc)
print("...Successfully encode sample using dictionary...")
print(100 * '-')


def encode(en_t, zh_t):
    # 因為字典的索引從 0 開始，
    # 我們可以使用 subword_encoder_en.vocab_size 這個值作為 BOS 的索引值
    # 用 subword_encoder_en.vocab_size + 1 作為 EOS 的索引值
    en_indices = [en_dict.vocab_size] + en_dict.encode(
        en_t.numpy()) + [en_dict.vocab_size + 1]
    # 同理，不過是使用中文字典的最後一個索引 + 1
    zh_indices = [zh_dict.vocab_size] + zh_dict.encode(
        zh_t.numpy()) + [zh_dict.vocab_size + 1]

    return en_indices, zh_indices


def tf_encode(en_t, zh_t):
    # 在 `tf_encode` 函式裡頭的 `en_t` 與 `zh_t` 都不是 Eager Tensors
    # 要到 `tf.py_funtion` 裡頭才是
    # 另外因為索引都是整數，所以使用 `tf.int64`
    return tf.py_function(encode, [en_t, zh_t], [tf.int64, tf.int64])


MAX_LENGTH = 40
BATCH_SIZE = 128
BUFFER_SIZE = 15000


def filter_max_length(en, zh, max_length=MAX_LENGTH):
    # en, zh 分別代表英文與中文的索引序列
    return tf.logical_and(tf.size(en) <= max_length,
                          tf.size(zh) <= max_length)


# 訓練集
train_dataset = (train_examples  # 輸出：(英文句子, 中文句子)
                 .map(tf_encode)  # 輸出：(英文索引序列, 中文索引序列)
                 .filter(filter_max_length)  # 同上，且序列長度都不超過 40
                 .cache()  # 加快讀取數據
                 .shuffle(BUFFER_SIZE)  # 將例子洗牌確保隨機性
                 .padded_batch(BATCH_SIZE,  # 將 batch 裡的序列都 pad 到一樣長度
                               padded_shapes=([40], [40]))
                 .prefetch(tf.data.experimental.AUTOTUNE))  # 加速
# 驗證集
test_dataset = (test_examples
                .map(tf_encode)
                .filter(filter_max_length)
                .padded_batch(BATCH_SIZE,
                              padded_shapes=([40], [40])))

print("re-construct the dataset:\nencoding with EOF,BOF;\nfilter those longer than 40;\npadding to 40 character each "
      "sentence")
en_batch, zh_batch = next(iter(train_dataset))
print(en_batch)
print(zh_batch)
print("...re-construct dataset completed...")
print(100 * '-')

# Actual trainning for Transformer
print("Actural trainning:")
print(20 * '-')
print("Hyper-Parameters:")
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = en_dict.vocab_size + 2
target_vocab_size = zh_dict.vocab_size + 2
dropout_rate = 0.1  # default value

# 論文裡頭最基本的 Transformer 配置為：
#
# num_layers=6
# d_model=512
# dff=2048

print(f"This transformer has {num_layers} of Encoder/Decoder layers:",
      "\nd_model:", d_model,
      "\nnum_heads", num_heads,
      "\ndff:", dff,
      "\nnum_heads:", num_heads,
      "\ninput_vocab_size:", input_vocab_size,
      "\ntarget_vocab_size:", target_vocab_size,
      "\ndropout:", dropout_rate,
      "\n")

print(20 * '-')
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate)

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
print(f"the last epochs: {min(0, last_epoch - EPOCHS)}")

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


# 給定一個英文句子，輸出預測的中文索引數字序列以及注意權重 dict
def evaluate(inp_sentence):
    # 準備英文句子前後會加上的 <start>, <end>
    start_token = [en_dict.vocab_size]
    end_token = [en_dict.vocab_size + 1]

    # inp_sentence 是字串，我們用 Subword Tokenizer 將其變成子詞的索引序列
    # 並在前後加上 BOS / EOS
    inp_sentence = start_token + en_dict.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # 跟我們在影片裡看到的一樣，Decoder 在第一個時間點吃進去的輸入
    # 是一個只包含一個中文 <start> token 的序列
    decoder_input = [zh_dict.vocab_size]
    output = tf.expand_dims(decoder_input, 0)  # 增加 batch 維度

    # auto-regressive，一次生成一個中文字並將預測加到輸入再度餵進 Transformer
    for i in range(MAX_LENGTH):
        # 每多一個生成的字就得產生新的遮罩
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # 將序列中最後一個 distribution 取出，並將裡頭值最大的當作模型最新的預測字
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 遇到 <end> token 就停止回傳，代表模型已經產生完結果
        if tf.equal(predicted_id, zh_dict.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        # 將 Transformer 新預測的中文索引加到輸出序列中，讓 Decoder 可以在產生
        # 下個中文字的時候關注到最新的 `predicted_id`
        output = tf.concat([output, predicted_id], axis=-1)

    # 將 batch 的維度去掉後回傳預測的中文索引序列
    return tf.squeeze(output, axis=0), attention_weights


def test():
    print("test")

# # 要被翻譯的英文句子
# sentence = "China, India, and others have enjoyed continuing economic growth."
#
# # 取得預測的中文索引序列
# predicted_seq, _ = evaluate(sentence)
#
# # 過濾掉 <start> & <end> tokens 並用中文的 subword tokenizer 幫我們將索引序列還原回中文句子
# target_vocab_size = zh_dict.vocab_size
# predicted_seq_without_bos_eos = [idx for idx in predicted_seq if idx < target_vocab_size]
# predicted_sentence = zh_dict.decode(predicted_seq_without_bos_eos)
#
# print("sentence:", sentence)
# print("-" * 20)
# print("predicted_seq:", predicted_seq)
# print("-" * 20)
# print("predicted_sentence:", predicted_sentence)
