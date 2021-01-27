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
import multiprocessing
import tensorflow as tf
import numpy as np

from ..asr_dataset import ASRDataset, AUTOTUNE, TFRECORD_SHARDS
from ..base_dataset import BUFFER_SIZE
from ...featurizers.speech_featurizers import SpeechFeaturizer, read_raw_audio, tf_read_raw_audio
from ...featurizers.text_featurizers import TextFeaturizer
from ...utils.utils import get_num_batches, bytestring_feature, print_one_line
from ...augmentations.augments import Augmentation


class ASRDatasetKeras(ASRDataset):
    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                {
                    "input": tf.TensorShape(self.speech_featurizer.shape),
                    "input_length": tf.TensorShape([]),
                    "prediction": tf.TensorShape([None]),
                    "prediction_length": tf.TensorShape([])
                },
                {
                    "label": tf.TensorShape([None]),
                    "label_length": tf.TensorShape([])
                },
            ),
            padding_values=(
                {
                    "input": 0.,
                    "input_length": 0,
                    "prediction": self.text_featurizer.blank,
                    "prediction_length": 0
                },
                {
                    "label": self.text_featurizer.blank,
                    "label_length": 0
                }
            ),
            drop_remainder=True
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(self.total_steps, batch_size)
        return dataset


class ASRTFRecordDatasetKeras(ASRDatasetKeras):
    """ Keras Dataset for ASR using TFRecords """

    def __init__(self,
                 data_paths: list,
                 tfrecords_dir: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 stage: str,
                 augmentations: Augmentation = Augmentation(None),
                 tfrecords_shards: int = TFRECORD_SHARDS,
                 cache: bool = False,
                 shuffle: bool = False,
                 buffer_size: int = BUFFER_SIZE):
        super(ASRTFRecordDatasetKeras, self).__init__(
            stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, augmentations=augmentations, cache=cache, shuffle=shuffle, buffer_size=buffer_size
        )
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0: raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

    def write_tfrecord_file(self, splitted_entries):
        shard_path, entries = splitted_entries
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for audio_file, _, transcript in entries:
                with tf.io.gfile.GFile(audio_file, mode="rb") as f:
                    audio = f.read()
                indices = " ".join([str(x) for x in self.text_featurizer.extract(transcript).numpy()])
                feature = {
                    "path": bytestring_feature([bytes(audio_file, "utf-8")]),
                    "audio": bytestring_feature([audio]),
                    "indices": bytestring_feature([bytes(indices, "utf-8")])
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                out.write(example.SerializeToString())
                print_one_line("Processed:", audio_file)
        print(f"\nCreated {shard_path}")

    def create_tfrecords(self):
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

        if tf.io.gfile.glob(os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")):
            print(f"TFRecords're already existed: {self.stage}")
            return True

        print(f"Creating {self.stage}.tfrecord ...")

        entries = self.read_entries()
        if len(entries) <= 0:
            return False

        def get_shard_path(shard_id):
            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_entries = np.array_split(entries, self.tfrecords_shards)
        with multiprocessing.Pool(self.tfrecords_shards) as pool:
            pool.map(self.write_tfrecord_file, zip(shards, splitted_entries))

        return True

    def preprocess(self, audio, indices):
        with tf.device("/CPU:0"):
            signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)

            signal = self.augmentations.before.augment(signal)

            features = self.speech_featurizer.extract(signal)

            features = self.augmentations.after.augment(features)

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
            label_length = tf.cast(tf.shape(label)[0], tf.int32)
            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)
            features = tf.convert_to_tensor(features, tf.float32)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            return features, input_length, label, label_length, prediction, prediction_length

    @tf.function
    def parse(self, record):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)

        features, input_length, label, label_length, \
            prediction, prediction_length = tf.numpy_function(
                self.preprocess,
                inp=[example["audio"], example["indices"]],
                Tout=[tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
            )

        return (
            {
                "input": features,
                "input_length": input_length,
                "prediction": prediction,
                "prediction_length": prediction_length
            },
            {
                "label": label,
                "label_length": label_length
            }
        )

    def create(self, batch_size):
        # Create TFRecords dataset
        have_data = self.create_tfrecords()
        if not have_data: return None

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)

        return self.process(dataset, batch_size)


class TFASRTFRecordDatasetKeras(ASRTFRecordDatasetKeras):
    def preprocess(self, audio, indices):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)

            signal = self.augmentations.before.augment(signal)

            features = self.speech_featurizer.tf_extract(signal)

            features = self.augmentations.after.augment(features)

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
            label_length = tf.cast(tf.shape(label)[0], tf.int32)
            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)
            features = tf.convert_to_tensor(features, tf.float32)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            return features, input_length, label, label_length, prediction, prediction_length

    @tf.function
    def parse(self, record):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)

        features, input_length, label, label_length, \
            prediction, prediction_length = self.preprocess([example["audio"], example["indices"]])

        return (
            {
                "input": features,
                "input_length": input_length,
                "prediction": prediction,
                "prediction_length": prediction_length
            },
            {
                "label": label,
                "label_length": label_length
            }
        )


class ASRSliceDatasetKeras(ASRDatasetKeras):
    """ Keras Dataset for ASR using Slice """

    def preprocess(self, path, transcript):
        return super(ASRSliceDatasetKeras, self).preprocess(path.decode("utf-8"), transcript)

    @tf.function
    def parse(self, record):
        features, input_length, label, label_length, \
            prediction, prediction_length = tf.numpy_function(
                self.preprocess,
                inp=[record[0], record[1]],
                Tout=[tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
            )
        return (
            {
                "input": features,
                "input_length": input_length,
                "prediction": prediction,
                "prediction_length": prediction_length
            },
            {
                "label": label,
                "label_length": label_length
            }
        )

    def create(self, batch_size):
        entries = self.read_entries()
        if len(entries) == 0: return None
        entries = np.delete(entries, 1, 1)  # Remove unused duration
        dataset = tf.data.Dataset.from_tensor_slices(entries)
        return self.process(dataset, batch_size)


class TFASRSliceDatasetKeras(ASRSliceDatasetKeras):
