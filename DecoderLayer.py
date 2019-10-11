import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention
from position_wise_feed_forward_network import point_wise_feed_forward_network


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder layer 用 MHA 1 來關注輸出序列，查詢 Q、鍵值 K 以及值 V 都是自己。
    而之所以有個 masked 是因為（中文）輸出序列除了跟（英文）輸入序列一樣需要 padding mask 以外，
    還需要 look ahead mask 來避免 Decoder layer 關注到未來的子詞.

    MHA1 處理完的輸出序列會成為 MHA 2 的 Q，而 K 與 V 則使用 Encoder 的輸出序列。
    這個運算的概念是讓一個 Decoder layer 在生成新的中文子詞時先參考先前已經產生的中文字，
    並為當下要生成的子詞產生一個包含前文語義的 repr. 。
    接著將此 repr. 拿去跟 Encoder 那邊的英文序列做匹配，看當下字詞的 repr. 有多好並予以修正。

    用簡單點的說法就是 Decoder 在生成中文字詞時除了參考已經生成的中文字以外，
    也會去關注 Encoder 輸出的英文子詞（的 repr.）。
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        # 3 個 sub-layers 的主角們
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # 定義每個 sub-layer 用的 LayerNorm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 定義每個 sub-layer 用的 Dropout
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, enc_output, training, combined_mask, en_padding_mask):
        # 所有 sub-layers 的主要輸出皆為 (batch_size, target_seq_len, d_model)
        # enc_output 為 Encoder 輸出序列，shape 為 (batch_size, input_seq_len, d_model)
        # attn_weights_block_1 則為 (batch_size, num_heads, target_seq_len, target_seq_len)
        # attn_weights_block_2 則為 (batch_size, num_heads, target_seq_len, input_seq_len)


        # sub-layer 1: decoder layer itself conduct self-attention to the output.
        # combined mask is the look_ahead_mask and the padding_mask for the output.
        # 來避免前面已生成的子詞關注到未來的子詞以及 <pad>
        _, _, _, attention_output1, _,attention_weights1, _ = self.mha1(x,x,x,combined_mask)
        attention_output1 = self.dropout1(attention_output1,training = training)
        output1 = self.layernorm1(attention_output1+x)

        # sub-layer 2: decoder layer attention to the output of the encoder.
        # 避免关注到encoder的<pad>
        _, _, _, attention_output2, _, attention_weights2, _ = self.mha2(enc_output, enc_output, output1, en_padding_mask)
        attention_output2 = self.dropout2(attention_output2,training = training)
        output2 = self.layernorm2(attention_output2 + output1)

        # sub-layer 3: FFN
        ffn_output = self.ffn(output2)
        ffn_output = self.dropout3(ffn_output,training = training)
        output3 = self.layernorm3(ffn_output + output2)

        return output3, attention_weights1, attention_weights2


