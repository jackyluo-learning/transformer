import tensorflow as tf


# 建立 Transformer 裡 Encoder / Decoder layer 都有使用到的 Feed Forward 元件
def point_wise_feed_forward_network(d_model, dff):
    # 此 FFN 對輸入做兩個線性轉換，中間加了一個 ReLU activation func
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff) output of the middle layer is dff
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

# 此函式在每次被呼叫的時候都會回傳一組新的全連接前饋神經網路（Fully-connected Feed Forward Network，FFN），其輸入張量與輸出張量的最後一個維度皆為 d_model，而在 FFN 中間層的維度則為
# dff。一般會讓 dff 大於 d_model，讓 FFN 從輸入的 d_model 維度裡頭擷取些有用的資訊。
