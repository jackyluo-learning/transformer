import tensorflow as tf


def split_heads(x, d_model, num_heads):
    # x.shape: (batch_size, seq_len, d_model)
    batch_size = tf.shape(x)[0]

    # 我們要確保維度 `d_model` 可以被平分成 `num_heads` 個 `depth` 維度
    assert d_model % num_heads == 0
    depth = d_model // num_heads  # 這是分成多頭以後每個向量的維度

    # 將最後一個 d_model 維度分成 num_heads 個 depth 維度。
    # 最後一個維度變成兩個維度，張量 x 從 3 維到 4 維
    # (batch_size, seq_len, num_heads, depth)
    reshaped_x = tf.reshape(x, shape=(batch_size, -1, num_heads, depth))

    # 將 head 的維度拉前使得最後兩個維度為子詞以及其對應的 depth 向量
    # (batch_size, num_heads, seq_len, depth)
    output = tf.transpose(reshaped_x, perm=[0, 2, 1, 3])

    return output