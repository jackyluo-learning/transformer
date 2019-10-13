import tensorflow as tf

from loss_object import loss_object


def loss_function(real, pred):
    # 這次的 mask 將序列中不等於 0 的位置視為 1，其餘為 0
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    print("mask:",mask)
    # 照樣計算所有位置的 cross entropy 但不加總
    loss_ = loss_object(real, pred)
    print("loss:",loss_)
    mask = tf.cast(mask, dtype=loss_.dtype)
    print("mask:",mask)
    loss_ *= mask  # 只計算非 <pad> 位置的損失
    print("loss:",loss_)

    return tf.reduce_mean(loss_)  # 将所有元素求平均，输出为一个数