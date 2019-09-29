import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import time
import os
import logging

# logging.basicConfig(level="ERROR")
output_dir = "nmt"  # where the generated dictionary will locate
en_vocab_file = os.path.join(output_dir, "en_vocab")  # join the two sub-directories
zh_vocab_file = os.path.join(output_dir, "zh_vocab")

def load_dictionary():
    # logging.basicConfig(level="DEBUG")
    start = time.process_time()
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_vocab_file)
    logging.info(f"loaded the English dictionary： {en_vocab_file}")
    logging.info(f"the size：{subword_encoder_en.vocab_size}")

    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
    logging.info(f"loaded the Chinese dictionary: {zh_vocab_file}")
    logging.info(f"the size: {subword_encoder_zh.vocab_size}")
    end = time.process_time()
    logging.info(f"process time: {end - start}")

    return subword_encoder_en, subword_encoder_zh