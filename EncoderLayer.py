import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention
from position_wise_feed_forward_network import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):  # rate here stands for the dropout rate.
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model,dff)

        # layer norm 很常在 RNN-based 的模型被使用。一個 sub-layer 一個 layer norm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 一樣，一個 sub-layer 一個 dropout layer
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training, mask):
        # 除了 `attn`，其他張量的 shape 皆為 (batch_size, input_seq_len, d_model)
        # attn.shape == (batch_size, num_heads, input_seq_len, input_seq_len)

        # sub-layer 1: MHA
        # Encoder 利用注意機制關注自己當前的序列，因此 v, k, q 全部都是自己
        # 另外別忘了我們還需要 padding mask 來遮住輸入序列中的 <pad> token
        _, _, _, attention_output, _, attention_weights,_ = self.mha(x,x,x,mask)
        attention_output = self.dropout1(attention_output,training = training)
        output1 = self.layernorm1(x + attention_output)  # residual connection

        # sub-layrer 2: FFN
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output,training = training)
        output2 = self.layernorm2(ffn_output + output1)

        return output2
