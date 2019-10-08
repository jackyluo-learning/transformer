import load_dictionary as ld
import load_dataset as lds
import tensorflow as tf

vq = tf.random.normal([2,3])
vk = tf.random.normal([1,3])
print(vq)
print(vk)
print(vq+vk)
# print(vq[:,0:-1])

