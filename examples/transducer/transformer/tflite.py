# Copyright 2020 Huy Le Nguyen (@nglehuy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import exec_helpers, featurizer_helpers
from tensorflow_asr.models.transducer.transformer import Transformer
from tensorflow_asr.utils import cli_util, env_util, file_util

logger = env_util.setup_environment()


def main(
    config_path: str,
    h5: str = None,
    output: str = None,
):
    assert h5 and output
    tf.keras.backend.clear_session()
    tf.compat.v1.enable_control_flow_v2()

    config = Config(config_path)
    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

    transformer = Transformer(**config.model_config, vocab_size=text_featurizer.num_classes)
    transformer.make(speech_featurizer.shape)
    transformer.load_weights(h5, by_name=file_util.is_hdf5_filepath(h5))
    transformer.summary()
    transformer.add_featurizers(speech_featurizer, text_featurizer)

    exec_helpers.convert_tflite(model=transformer, output=output)


if __name__ == "__main__":
    cli_util.run(main)
