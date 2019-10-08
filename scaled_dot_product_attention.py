import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    # 將 `q`、 `k` 做點積再 scale
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)  #多維張量相乘，前面的維度當成batch，後面兩個維度相同即可

    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # 取得 seq_k 的序列長度
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # scale by sqrt(dk)

    # 將遮罩「加」到被丟入 softmax 前的 logits
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 取 softmax 是為了得到總和為 1 的比例之後對 `v` 做加權平均
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # 以注意權重對 v 做加權平均（weighted average）
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights, scaled_attention_logits


