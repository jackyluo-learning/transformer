import tensorflow as tf

from Encoder import Encoder
from Decoder import Decoder
from create_masks import create_look_ahead_mask
from create_masks import create_padding_mask


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        # this fnn output the same number of logits as the target vocab size, as it represent as the probabilities of
        # the occurrence of every target word

    @tf.function
    def __call__(self, input, target, training):
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
            input, target)

        enc_output = self.encoder(x=input, mask=enc_padding_mask, training=training)

        dec_output, attention_weights = self.decoder(target, enc_output, training, combined_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # Decoder 的輸出 dec_output 則會通過 Final linear layer，被轉成進入 Softmax 前的 logits final_output，
        # 其 logit 的數目則跟中文字典裡的子詞數相同。
        # 把英文（來源）以及中文（目標）的索引序列 batch 丟入 Transformer，
        # 它就會輸出最後一維為中文字典大小的張量。

        return final_output, attention_weights

    def create_masks(self, inp, tar):
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
