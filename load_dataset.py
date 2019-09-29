import tensorflow_datasets as tfds
import logging
from pprint import pprint

dataset = tfds
def load_dataset():
    # logging.basicConfig(level=logging.ERROR)
    downloaded_dir = "tensorflow-datasets/downloads"
    config = tfds.translate.wmt.WmtConfig(
        version=tfds.core.Version('0.0.3', experiments={tfds.core.Experiment.S3: False}),
        language_pair=("zh", "en"),
        subsets={
            tfds.Split.TRAIN: ["newscommentary_v14"]  # select the news comment train dataset to be the dataset,
        }
    )
    train_perc = 20
    val_prec = 1
    drop_prec = 100 - train_perc - val_prec

    builder = tfds.builder("wmt_translate",config=config)
    builder.download_and_prepare(download_dir=downloaded_dir)
    split = tfds.Split.TRAIN.subsplit([train_perc, val_prec, drop_prec])
    dataset = builder.as_dataset(split=split,as_supervised=True)
    return dataset
# logging.basicConfig(level="DEBUG")
# logging.info(dataset)
