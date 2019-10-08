import tensorflow as tf
from scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0  # sure that the d_model can be divided

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)  # 分別給 q, k, v 的 3 個線性轉換
        self.wk = tf.keras.layers.Dense(d_model)  # 注意我們並沒有指定 activation func
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self,x,batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,v,k,q,mask):
        batch_size = tf.shape(q)[0]

        # 將輸入的 q, k, v 都各自做一次線性轉換到 `d_model` 維空間
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 前面看過的，將最後一個 `d_model` 維度分成 `num_heads` 個 `depth` 維度
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 利用 broadcasting 讓每個句子的每個 head 的 qi, ki, vi 都各自進行注意力機制
        # 輸出會多一個 head 維度
        scaled_attention, attention_weights, scaled_attention_logits = scaled_dot_product_attention(
            q, k, v, mask)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 跟我們在 `split_heads` 函式做的事情剛好相反，先做 transpose 再做 reshape
        # 將 `num_heads` 個 `depth` 維度串接回原來的 `d_model` 維度
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        # 通過最後一個線性轉換
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return q, k, v, output, scaled_attention, attention_weights, scaled_attention_logits