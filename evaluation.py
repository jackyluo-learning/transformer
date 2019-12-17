import logging
import os
import re

import tensorflow as tf

from CustomSchedule import CustomSchedule
from hparams import basic_param
from Transformer import Transformer
import matplotlib.pyplot as plt
from load_dictionary import load_dictionary
from load_dataset import load_dataset
from nltk.translate.bleu_score import sentence_bleu

logging.basicConfig(level=logging.ERROR)


def evaluate(inp_sentence, en_dict, zh_dict, transformer, max_length):
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
    for i in range(max_length):
        # 每多一個生成的字就得產生新的遮罩

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer.__call__(encoder_input,
                                                              output,
                                                              False,
                                                              )

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


def plot_attention_weights(attention, sentence, result, layer_name, en_dict, zh_dict):
    fig = plt.figure(figsize=(16, 8))

    sentence = en_dict.encode(sentence)

    attention = tf.squeeze(attention[layer_name], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # 画出注意力权重
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [en_dict.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([zh_dict.decode([i]) for i in result
                            if i < zh_dict.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(sentence, en_dict, zh_dict, transformer, max_length, plot='decoder_layer4_dec_enc'):
    result, attention_weights = evaluate(sentence, en_dict, zh_dict, transformer, max_length)
    print("attention_weights.keys():")
    for layer_name, attn in attention_weights.items():
        print(f"{layer_name}.shape: {attn.shape}")
    predicted_sentence = zh_dict.decode([i for i in result
                                         if i < zh_dict.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot, en_dict, zh_dict)


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


def filter_max_length(en, zh, max_length=40):
    # en, zh 分別代表英文與中文的索引序列
    return tf.logical_and(tf.size(en) <= max_length,
                          tf.size(zh) <= max_length)


if __name__ == '__main__':
    p = basic_param()
    train_examples, test_examples, _ = load_dataset()
    test_dataset = (test_examples
                    .map(tf_encode)
                    .filter(filter_max_length)
                    )
    en_dict, zh_dict = load_dictionary()
    output_dir = "nmt"
    checkpoint_path = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, 'logs')
    # train_perc = 20
    run_id = f"{p.num_layers}layers_{p.d_model}d_{p.num_heads}heads_{p.dff}dff_{p.train_perc}train_perc"
    checkpoint_path = os.path.join(checkpoint_path, run_id)
    # log_dir = os.path.join(log_dir, run_id)
    learning_rate = CustomSchedule(p.d_model)

    p.add_param('input_vocab_size', 8137)
    p.add_param('target_vocab_size', 4203)
    p.add_param('learning_rate', learning_rate)
    transformer = Transformer(p.num_layers, p.d_model, p.num_heads, p.dff, p.input_vocab_size, p.target_vocab_size,
                              p.dropout_rate)
    optimizer = tf.keras.optimizers.Adam(p.learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    # tf.train.Checkpoint 可以幫我們把想要存下來的東西整合起來，方便儲存與讀取
    # 一般來說你會想存下模型以及 optimizer 的狀態
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        print("exist")
        ckpt.restore(ckpt_manager.latest_checkpoint)
        last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        print(f'已讀取最新的 checkpoint，模型已訓練 {last_epoch} epochs。')
    # transformer = tf.saved_model.load('models/transformer_base')
    print("...Model loaded...")
    print('-' * 20)
    # translate("China, India, and others have enjoyed continuing growth.", en_dict, zh_dict, transformer,
    #           p.max_length)
    score_list = []
    for (x, (inp, tar)) in enumerate(test_dataset):
        # for x in range(p.batch_size):
            print('batch id:', x)
            sentence = en_dict.decode([i for i in inp if i < en_dict.vocab_size])
            print('English sentence: ',sentence)
            print('Encoded input:', inp)
            print('Real output: ',tar)
            result, _ = evaluate(sentence, en_dict, zh_dict, transformer, p.max_length)
            print('Prediction: ',result)
            print('Prediction sentence: ', zh_dict.decode([i for i in result if i < zh_dict.vocab_size]) )
            target = tar.numpy().tolist()
            result = result.numpy().tolist()
            # print(tar)
            # print(result)
            score = sentence_bleu([target], result)
            print('BLEU: ',score)
            score_list.append(score)

    print(sum(score_list/len(score_list)))




    # ckpt_manager 會去 checkpoint_path 看有沒有符合 ckpt 裡頭定義的東西
    # 存檔的時候只保留最近 5 次 checkpoints，其他自動刪除
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
