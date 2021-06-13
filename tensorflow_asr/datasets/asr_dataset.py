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
import json
from typing import Union
import tqdm
import numpy as np
import tensorflow as tf

from ..augmentations.augmentation import Augmentation
from .base_dataset import BaseDataset, BUFFER_SIZE, TFRECORD_SHARDS, AUTOTUNE
from ..featurizers.speech_featurizers import load_and_convert_to_wav, read_raw_audio, tf_read_raw_audio, SpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils import feature_util, file_util, math_util, data_util

logger = tf.get_logger()


class ASRDataset(BaseDataset):
    """ Dataset for ASR using Generator """

    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: Augmentation = Augmentation(None),
                 cache: bool = False,
                 shuffle: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 use_tf: bool = False,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        super().__init__(
            data_paths=data_paths, augmentations=augmentations,
            cache=cache, shuffle=shuffle, stage=stage, buffer_size=buffer_size,
            drop_remainder=drop_remainder, use_tf=use_tf, indefinite=indefinite
        )
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    # -------------------------------- metadata -------------------------------------

    def compute_metadata(self):
        self.read_entries()
        for _, duration, indices in tqdm.tqdm(self.entries, desc=f"Computing metadata for entries in {self.stage} dataset"):
            input_length = self.speech_featurizer.get_length_from_duration(duration)
            label = str(indices).split()
            label_length = len(label)
            self.speech_featurizer.update_length(input_length)
            self.text_featurizer.update_length(label_length)

    def save_metadata(self, metadata: str = None):
        if metadata is None: return
        metadata = file_util.preprocess_paths(metadata)
        if tf.io.gfile.exists(metadata):
            with tf.io.gfile.GFile(metadata, "r") as f:
                try:
                    content = json.loads(f.read())
                except json.JSONDecodeError:
                    raise ValueError(f'File {metadata} is currently not in json format. Please update the file')
        else:
            content = {}
        content[self.stage] = {
            "max_input_length": self.speech_featurizer.max_length,
            "max_label_length": self.text_featurizer.max_length,
            "num_entries": self.total_steps
        }
        with tf.io.gfile.GFile(metadata, "w") as f:
            f.write(json.dumps(content, indent=2))
        logger.info(f"Metadata written to {metadata}")

    def load_metadata(self, metadata: Union[str, dict] = None):
        if metadata is None: return
        content = None
        if isinstance(metadata, dict):
            content = metadata
        else:
            metadata = file_util.preprocess_paths(metadata)
            if tf.io.gfile.exists(metadata):
                logger.info(f"Loading metadata from {metadata} ...")
                with tf.io.gfile.GFile(metadata, "r") as f:
                    try:
                        content = json.loads(f.read()).get(self.stage, {})
                    except json.JSONDecodeError:
                        raise ValueError(f'File {metadata} must be in json format')
        if not content:
            return
        self.speech_featurizer.update_length(int(content.get("max_input_length", 0)))
        self.text_featurizer.update_length(int(content.get("max_label_length", 0)))
        self.total_steps = int(content.get("num_entries", 0))

    def update_metadata(self, metadata: str = None):
        self.load_metadata(metadata)
        self.compute_metadata()
        self.save_metadata(metadata)

    # -------------------------------- ENTRIES -------------------------------------

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0: return
        self.entries = []
        for file_path in self.data_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                self.entries += temp_lines[1:]
        # The files is "\t" seperated
        self.entries = [line.split("\t", 2) for line in self.entries]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1]).numpy()])
        self.entries = np.array(self.entries)
        if self.shuffle: np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)

    # -------------------------------- LOAD AND PREPROCESS -------------------------------------

    def generator(self):
        for path, _, indices in self.entries:
            audio = load_and_convert_to_wav(path).numpy()
            yield bytes(path, "utf-8"), audio, bytes(indices, "utf-8")

    def preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            def fn(_path: bytes, _audio: bytes, _indices: bytes):
                signal = read_raw_audio(_audio, sample_rate=self.speech_featurizer.sample_rate)
                signal = self.augmentations.signal_augment(signal)
                features = self.speech_featurizer.extract(signal.numpy())
                features = self.augmentations.feature_augment(features)
                features = tf.convert_to_tensor(features, tf.float32)
                input_length = tf.cast(tf.shape(features)[0], tf.int32)

                label = tf.strings.to_number(tf.strings.split(_indices), out_type=tf.int32)
                label_length = tf.cast(tf.shape(label)[0], tf.int32)

                prediction = self.text_featurizer.prepand_blank(label)
                prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

                return _path, features, input_length, label, label_length, prediction, prediction_length

            return tf.numpy_function(
                fn, inp=[path, audio, indices],
                Tout=[tf.string, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
            )

    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)
            signal = self.augmentations.signal_augment(signal)
            features = self.speech_featurizer.tf_extract(signal)
            features = self.augmentations.feature_augment(features)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
            label_length = tf.cast(tf.shape(label)[0], tf.int32)

            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

            return path, features, input_length, label, label_length, prediction, prediction_length

    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        data = self.tf_preprocess(path, audio, indices) if self.use_tf else self.preprocess(path, audio, indices)
        _, features, input_length, label, label_length, prediction, prediction_length = data
        return (
            data_util.create_inputs(
                inputs=features,
                inputs_length=input_length,
                predictions=prediction,
                predictions_length=prediction_length
            ),
            data_util.create_labels(
                labels=label,
                labels_length=label_length
            )
        )

    # -------------------------------- CREATION -------------------------------------

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        self.total_steps = math_util.get_num_batches(self.total_steps, batch_size, drop_remainders=self.drop_remainder)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                data_util.create_inputs(
                    inputs=tf.TensorShape(self.speech_featurizer.shape),
                    inputs_length=tf.TensorShape([]),
                    predictions=tf.TensorShape(self.text_featurizer.prepand_shape),
                    predictions_length=tf.TensorShape([])
                ),
                data_util.create_labels(
                    labels=tf.TensorShape(self.text_featurizer.shape),
                    labels_length=tf.TensorShape([])
                ),
            ),
            padding_values=(
                data_util.create_inputs(
                    inputs=0.0,
                    inputs_length=0,
                    predictions=self.text_featurizer.blank,
                    predictions_length=0
                ),
                data_util.create_labels(
                    labels=self.text_featurizer.blank,
                    labels_length=0
                )
            ),
            drop_remainder=self.drop_remainder
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
        )
        return self.process(dataset, batch_size)


class ASRTFRecordDataset(ASRDataset):
    """ Dataset for ASR using TFRecords """

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
                 use_tf: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        super().__init__(
            stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, augmentations=augmentations, cache=cache, shuffle=shuffle, buffer_size=buffer_size,
            drop_remainder=drop_remainder, use_tf=use_tf, indefinite=indefinite
        )
        if not self.stage:
            raise ValueError("stage must be defined, either 'train', 'eval' or 'test'")
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0:
            raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards

    @staticmethod
    def write_tfrecord_file(splitted_entries):
        shard_path, entries = splitted_entries

        def parse(record):
            def fn(path, indices):
                audio = load_and_convert_to_wav(path.decode("utf-8")).numpy()
                feature = {
                    "path": feature_util.bytestring_feature([path]),
                    "audio": feature_util.bytestring_feature([audio]),
                    "indices": feature_util.bytestring_feature([indices])
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                return example.SerializeToString()
            return tf.numpy_function(fn, inp=[record[0], record[2]], Tout=tf.string)

        dataset = tf.data.Dataset.from_tensor_slices(entries)
        dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)
        writer = tf.data.experimental.TFRecordWriter(shard_path, compression_type="ZLIB")
        logger.info(f"Processing {shard_path} ...")
        writer.write(dataset)
        logger.info(f"Created {shard_path}")

    def create_tfrecords(self):
        if not self.tfrecords_dir:
            return False

        if tf.io.gfile.glob(os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")):
            logger.info(f"TFRecords're already existed: {self.stage}")
            return True

        logger.info(f"Creating {self.stage}.tfrecord ...")

        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return False

        def get_shard_path(shard_id):
            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_entries = np.array_split(self.entries, self.tfrecords_shards)
        for entries in zip(shards, splitted_entries):
            self.write_tfrecord_file(entries)

        return True

    def parse(self, record: tf.Tensor):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        return super().parse(**example)

    def create(self, batch_size: int):
        have_data = self.create_tfrecords()
        if not have_data: return None

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type="ZLIB", num_parallel_reads=AUTOTUNE)

        return self.process(dataset, batch_size)


class ASRSliceDataset(ASRDataset):
    """ Dataset for ASR using Slice """

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes): return load_and_convert_to_wav(path.decode("utf-8")).numpy()
        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)
        return record[0], audio, record[2]

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)


class ASRMaskedSliceDataset(ASRSliceDataset):
    """ Dataset for ASR with rolling mask """

    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: Augmentation = Augmentation(None),
                 cache: bool = False,
                 shuffle: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 use_tf: bool = False,
                 buffer_size: int = BUFFER_SIZE,
                 history_window_size: int = 3,
                 input_chunk_duration: int = 250,
                 time_reduction_factor: int = 4,
                 **kwargs):
        super(ASRMaskedSliceDataset, self).__init__(
            data_paths=data_paths, augmentations=augmentations,
            cache=cache, shuffle=shuffle, stage=stage, buffer_size=buffer_size,
            drop_remainder=drop_remainder, use_tf=use_tf, indefinite=indefinite,
            speech_featurizer=speech_featurizer, text_featurizer=text_featurizer
        )
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.history_window_size = history_window_size
        self.input_chunk_size = input_chunk_duration * self.speech_featurizer.sample_rate // 1000
        self.time_reduction_factor = time_reduction_factor

        self.base_mask = tf.constant(0, dtype=tf.int32)
        self.use_base_mask = False

        # If max input length is known, pre-compute mask
        if self.speech_featurizer.max_length:
            num_frames = 1 + (self.speech_featurizer.max_length - self.speech_featurizer.frame_length) // self.speech_featurizer.frame_step
            self.base_mask = self.calculate_mask(num_frames)
            self.use_base_mask = True

    def _recalculate_mask(self, num_frames):
        frames_per_chunk = self.input_chunk_size // self.speech_featurizer.frame_step

        def _calculate_mask(num_frames, frames_per_chunk, history_window_size):
            mask = np.zeros((num_frames, num_frames), dtype=np.int32)
            for i in range(num_frames):
                # Frames in the same chunk can see each other
                # If frames in `history_window_size` are in other chunks, the full chunks are visible
                current_chunk_index = i // frames_per_chunk
                history_chunk_index = (i - history_window_size) // frames_per_chunk
                for curr in range(history_chunk_index, current_chunk_index + 1):
                    for j in range(frames_per_chunk):
                        base_index = curr * frames_per_chunk
                        if base_index + j < 0 or base_index + j >= num_frames:
                            continue
                        mask[i, base_index + j] = 1
            return mask

        @tf.function(autograph=True)
        def _calculate_mask_tf(num_frames, frames_per_chunk, history_window_size):
            chunk_ids = tf.range(num_frames) // frames_per_chunk
            num_chunks = tf.cast(tf.math.ceil(num_frames / frames_per_chunk), dtype=tf.int32)

            # Create first `frames_per_chunk` rows
            current = tf.ones((frames_per_chunk), dtype=tf.int32)
            trailing = tf.ones(((num_chunks - 1) * frames_per_chunk), dtype=tf.int32)
            tmp_row = tf.concat((current, trailing), axis=0)
            row = tf.slice(tmp_row, [0], [num_frames])
            mask = tf.expand_dims(row, axis=0)

            for i in range(1, frames_per_chunk):
                mask = tf.concat((mask, [row]), axis=0)

            # Create the following rows
            for i in range(frames_per_chunk, num_frames):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(mask, tf.TensorShape([None, None]))]
                )
                curr_chunk_id = chunk_ids[i]
                hist_i = tf.math.maximum(i - history_window_size, 0)
                hist_chunk_id = chunk_ids[hist_i]

                # Build the left-most part
                leading_chunk_id = hist_chunk_id - 1
                num_leading_chunks = tf.math.maximum(leading_chunk_id + 1, 0)
                leftmost_row = tf.zeros((num_leading_chunks * frames_per_chunk), dtype=tf.int32)

                # Build the current visible chunks
                num_hist_chunks = curr_chunk_id - hist_chunk_id
                num_visible_chunks = num_hist_chunks + 1
                curr_chunk_row = tf.ones((num_visible_chunks * frames_per_chunk), dtype=tf.int32)

                # Build the trailing 0s
                num_trailing_chunks = tf.math.maximum((num_chunks - curr_chunk_id) - 1, 0)
                trailing_chunk_row = tf.zeros((num_trailing_chunks * frames_per_chunk), dtype=tf.int32)

                # Merge chunks, clip to output size
                tmp_row = tf.concat([leftmost_row, curr_chunk_row, trailing_chunk_row], axis=0)
                row = tf.slice(tmp_row, [0], [num_frames])

                mask = tf.concat((mask, [row]), axis=0)
            return mask

        if self.use_tf:
            mask = _calculate_mask_tf(num_frames, frames_per_chunk, self.history_window_size)
        else:
            mask = tf.numpy_function(
                _calculate_mask, inp=[num_frames, frames_per_chunk, self.history_window_size], Tout=tf.int32
            )
            mask.set_shape((None, None))

        return mask

    def calculate_mask(self, num_frames):
        num_frames = math_util.get_reduced_length(num_frames, self.time_reduction_factor)

        if self.use_base_mask:
            return tf.slice(self.base_mask, [0, 0], [num_frames, num_frames])
        return self._recalculate_mask(num_frames)

    def preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        preprocessed_inputs = super(ASRMaskedSliceDataset, self).preprocess(path, audio, indices)

        input_length = preprocessed_inputs[2]
        mask = self.calculate_mask(input_length)

        return (*preprocessed_inputs, mask)

    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        preprocessed_inputs = super(ASRMaskedSliceDataset, self).tf_preprocess(path, audio, indices)

        input_length = preprocessed_inputs[2]
        mask = self.calculate_mask(input_length)

        return (*preprocessed_inputs, mask)

    # -------------------------------- CREATION -------------------------------------

    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp, mask
        """
        if self.use_tf: data = self.tf_preprocess(path, audio, indices)
        else: data = self.preprocess(path, audio, indices)

        _, features, input_length, label, label_length, prediction, prediction_length, mask = data

        return (
            data_util.create_inputs(
                inputs=features,
                inputs_length=input_length,
                predictions=prediction,
                predictions_length=prediction_length,
                mask=mask
            ),
            data_util.create_labels(
                labels=label,
                labels_length=label_length
            )
        )

    def process(self, dataset: tf.data.Dataset, batch_size: int):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        self.total_steps = math_util.get_num_batches(self.total_steps, batch_size, drop_remainders=self.drop_remainder)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                data_util.create_inputs(
                    inputs=tf.TensorShape(self.speech_featurizer.shape),
                    inputs_length=tf.TensorShape([]),
                    predictions=tf.TensorShape(self.text_featurizer.prepand_shape),
                    predictions_length=tf.TensorShape([]),
                    mask=tf.TensorShape([self.speech_featurizer.shape[0], self.speech_featurizer.shape[0]])
                ),
                data_util.create_labels(
                    labels=tf.TensorShape(self.text_featurizer.shape),
                    labels_length=tf.TensorShape([])
                ),
            ),
            padding_values=(
                data_util.create_inputs(
                    inputs= 0.,
                    inputs_length=0,
                    predictions=self.text_featurizer.blank,
                    predictions_length=0,
                    mask=0
                ),
                data_util.create_labels(
                    labels=self.text_featurizer.blank,
                    labels_length=0
                )
            ),
            drop_remainder = self.drop_remainder
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
        )
        return self.process(dataset, batch_size)


class ASRMaskedTFRecordDataset(ASRMaskedSliceDataset, ASRTFRecordDataset):
    """ Dataset for ASR using TFRecords with rolling mask """
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
                 use_tf: bool = False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 buffer_size: int = BUFFER_SIZE,
                 history_window_size: int = 3,
                 input_chunk_duration: int = 250,
                 time_reduction_factor: int = 4,
                 **kwargs):
        super(ASRMaskedTFRecordDataset, self).__init__(
            stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, augmentations=augmentations, cache=cache, shuffle=shuffle, buffer_size=buffer_size,
            drop_remainder=drop_remainder, use_tf=use_tf, indefinite=indefinite, history_window_size = history_window_size,
            input_chunk_duration = input_chunk_duration, time_reduction_factor = time_reduction_factor,
        )
        if not self.stage: raise ValueError("stage must be defined, either 'train', 'eval' or 'test'")
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0: raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards
        if not tf.io.gfile.exists(self.tfrecords_dir): tf.io.gfile.makedirs(self.tfrecords_dir)

    def parse(self, record: tf.Tensor):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        if self.use_tf: data = self.tf_preprocess(**example)
        else: data = self.preprocess(**example)

        _, features, input_length, label, label_length, prediction, prediction_length, mask = data

        return (
            data_util.create_inputs(
                inputs=features,
                inputs_length=input_length,
                predictions=prediction,
                predictions_length=prediction_length,
                mask=mask
            ),
            data_util.create_labels(
                labels=label,
                labels_length=label_length
            )
        )
