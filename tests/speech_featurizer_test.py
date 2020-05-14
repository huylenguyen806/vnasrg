from __future__ import absolute_import, print_function

import os.path as o
import sys
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
from featurizers.SpeechFeaturizer import read_raw_audio, speech_feature_extraction
import matplotlib.pyplot as plt
import numpy as np


def main(argv):
  speech_file = argv[1]
  feature_type = argv[2]
  speech_conf = {
    "sample_rate": 16000,
    "frame_ms": 20,
    "stride_ms": 10,
    "feature_type": feature_type,
    "pre_emph": 0.97,
    "normalize_signal": True,
    "normalize_feature": True,
    "norm_per_feature": False,
    "num_feature_bins": int(16000 * 20 / 1000) // 2 + 1,
    "delta": True,
    "delta_delta": True,
    "pitch": True
  }
  signal = read_raw_audio(speech_file, speech_conf["sample_rate"])
  ft = speech_feature_extraction(signal, speech_conf)

  ftypes = [feature_type, "delta", "delta_delta", "pitch"]

  plt.figure(figsize=(15, 5))
  for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(ft[:, :, i].T, origin="lower")
    plt.title(ftypes[i])
    plt.colorbar()
    plt.tight_layout()
  plt.savefig(argv[3])
  plt.show()


if __name__ == "__main__":
  main(sys.argv)
