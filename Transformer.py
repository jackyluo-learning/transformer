import tensorflow as tf

from Encoder import Encoder
from Decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers,d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate = 0.1):
        super(Transformer,self).__init__()

        self.encoder = Encoder(num_layers,d_model,num_heads,dff,input_vocab_size,rate)

        self.decoder = Decoder(num_layers,d_model,num_heads,dff,target_vocab_size,rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        # this fnn output the same number of logits as the target vocab size, as it represent as the probabilities of
        # the occurrence of every target word


    def __call__(self, input, target, training, enc_padding_mask, combined_mask, dec_padding_mask):
        enc_output = self.encoder(x=input, mask = enc_padding_mask, training=training)

        dec_output, attention_weights = self.decoder(target,enc_output,training,combined_mask,dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # Decoder 的輸出 dec_output 則會通過 Final linear layer，被轉成進入 Softmax 前的 logits final_output，
        # 其 logit 的數目則跟中文字典裡的子詞數相同。
        # 把英文（來源）以及中文（目標）的索引序列 batch 丟入 Transformer，
        # 它就會輸出最後一維為中文字典大小的張量。

        return final_output,attention_weights

