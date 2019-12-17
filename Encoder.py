import tensorflow as tf
from positional_encoding import positional_encoding
from EncoderLayer import EncoderLayer


class Encoder(tf.keras.layers.Layer):
    # Encoder 的初始參數除了本來就要給 EncoderLayer 的參數還多了：
    # - num_layers: 決定要有幾個 EncoderLayers, 前面影片中的 `N`
    # - input_vocab_size: 用來把索引轉成詞嵌入向量
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embeding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training, mask):
        input_seq_len = tf.shape(x)[1]

        x = self.embeding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]

        # regularization
        x = self.dropout(x, training=training)

        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, training, mask)

        return x
