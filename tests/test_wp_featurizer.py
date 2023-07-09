# pylint: disable=line-too-long
import os

import tensorflow as tf

from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.tokenizers import WordPieceTokenizer
from tensorflow_asr.utils import file_util

file_util.ENABLE_PATH_PREPROCESS = False

config_path = os.path.join(file_util.ROOT_DIRECTORY, "examples", "configs", "wp_whitespace.yml.j2")
print(config_path)
config = file_util.load_yaml(config_path)

decoder_config = DecoderConfig(config["decoder_config"])

text = "<pad> i'm good but it would have broken down after ten miles of that hard trail dawn came while they wound over the crest of the range and with the sun in their faces they took the downgrade it was well into the morning before nash reached logan"
text = "a b"


def test_wordpiece_featurizer():
    featurizer = WordPieceTokenizer(decoder_config=decoder_config)
    print(featurizer.num_classes)
    print(text)
    indices = featurizer.tokenize(text)
    print(indices.numpy())
    batch_indices = tf.stack([indices, indices], axis=0)
    reversed_text = featurizer.detokenize(batch_indices)
    print(reversed_text.numpy())
    upoints = featurizer.detokenize_unicode_points(indices)
    print(upoints.numpy())
