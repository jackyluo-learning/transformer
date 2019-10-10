import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)

# model = tf.keras.Sequential()
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# 以下代码使用构造函数参数实例化 tf.keras.layers. Dense 层：
# Create a sigmoid layer:
# layers.Dense(64, activation='sigmoid')
# # Or:
# # layers.Dense(64, activation=tf.sigmoid)
#
# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
# layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
#
# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
#
# # A linear layer with a kernel initialized to a random orthogonal matrix:
# layers.Dense(64, kernel_initializer='orthogonal')
#
# # A linear layer with a bias vector initialized to 2.0s:
# layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])


data = np.random.random((1000, 64))
labels = np.random.random((1000, 10))
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(100)
dataset = dataset.repeat()

model.fit(dataset, epochs=10,steps_per_epoch=30)
