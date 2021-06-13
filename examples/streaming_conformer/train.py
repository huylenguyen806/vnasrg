# Copyright 2020 Huy Le Nguyen (@usimarit)
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

import os
import math
import argparse
from tensorflow_asr.utils import env_util

env_util.setup_environment()
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Conformer Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--max_ckpts", type=int, default=10, help="Max number of checkpoints to keep")

parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--subwords", default=False, action="store_true", help="Use subwords")

parser.add_argument("--tbs", type=int, default=None, help="Train batch size per replica")

parser.add_argument("--ebs", type=int, default=None, help="Evaluation batch size per replica")

parser.add_argument("--spx", type=int, default=1, help="Steps per execution for maximizing performance")

parser.add_argument("--metadata", type=str, default=None, help="Path to file containing metadata")

parser.add_argument("--static_length", default=False, action="store_true", help="Use static lengths")

parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

strategy = env_util.setup_strategy(args.devices)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRMaskedSliceDataset, ASRMaskedTFRecordDataset
from tensorflow_asr.featurizers import speech_featurizers, text_featurizers
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.models.transducer.streaming_conformer import StreamingConformer
from tensorflow_asr.optimizers.schedules import TransformerSchedule

config = Config(args.config)
speech_featurizer = speech_featurizers.TFSpeechFeaturizer(config.speech_config)

if args.sentence_piece:
    print("Loading SentencePiece model ...")
    text_featurizer = text_featurizers.SentencePieceFeaturizer(config.decoder_config)
elif args.subwords:
    print("Loading subwords ...")
    text_featurizer = text_featurizers.SubwordFeaturizer(config.decoder_config)
else:
    print("Use characters ...")
    text_featurizer = text_featurizers.CharFeaturizer(config.decoder_config)

time_reduction_factor = config.model_config['encoder_subsampling']['strides'] * 2
if args.tfrecords:
    train_dataset = ASRMaskedTFRecordDataset(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        **vars(config.learning_config.train_dataset_config)
    )
    eval_dataset = ASRMaskedTFRecordDataset(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        **vars(config.learning_config.eval_dataset_config)
    )
else:
    train_dataset = ASRMaskedSliceDataset(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        time_reduction_factor=time_reduction_factor,
        **vars(config.learning_config.train_dataset_config)
    )
    eval_dataset = ASRMaskedSliceDataset(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        time_reduction_factor=time_reduction_factor,
        **vars(config.learning_config.eval_dataset_config)
    )

train_dataset.load_metadata(args.metadata)
eval_dataset.load_metadata(args.metadata)

if not args.static_length:
    speech_featurizer.reset_length()
    text_featurizer.reset_length()

global_batch_size = args.tbs or config.learning_config.running_config.batch_size
global_batch_size *= strategy.num_replicas_in_sync

global_eval_batch_size = args.ebs or global_batch_size
global_eval_batch_size *= strategy.num_replicas_in_sync

train_data_loader = train_dataset.create(global_batch_size)
eval_data_loader = eval_dataset.create(global_eval_batch_size)

with strategy.scope():
    # build model
    streaming_conformer = StreamingConformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    streaming_conformer.make(speech_featurizer.shape)
    streaming_conformer.summary(line_length=150)

    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=streaming_conformer.dmodel,
            warmup_steps=config.learning_config.optimizer_config.pop("warmup_steps", 10000),
            max_lr=(0.05 / math.sqrt(streaming_conformer.dmodel))
        ),
        **config.learning_config.optimizer_config
    )

    streaming_conformer.compile(
        optimizer=optimizer,
        experimental_steps_per_execution=args.spx,
        global_batch_size=global_batch_size,
        blank=text_featurizer.blank
    )

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
    tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
    tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard)
]

streaming_conformer.fit(
    train_data_loader,
    batch_size=global_batch_size,
    epochs=config.learning_config.running_config.num_epochs,
    steps_per_epoch=train_dataset.total_steps,
    validation_data=eval_data_loader,
    validation_batch_size=global_eval_batch_size,
    validation_steps=eval_dataset.total_steps,
    callbacks=callbacks,
)
