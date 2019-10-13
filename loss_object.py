import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# 检查 y_true 中的值（本身就是index） 与 y_pred 中最大值对应的index是否相等。