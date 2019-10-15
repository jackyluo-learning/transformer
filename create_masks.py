import tensorflow as tf


# padding mask: mask those pad with 0
def create_padding_mask(seq):
    # padding mask 的工作就是把索引序列中為 0 的位置設為 1
    mask = tf.cast(tf.equal(seq, 0), tf.float32)  # tf.equal(): compare each in seq to 0, equal return True.
    return mask[:, tf.newaxis, tf.newaxis, :]  # broadcasting


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
