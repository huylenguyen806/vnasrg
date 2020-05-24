from __future__ import absolute_import

import os
import sys
import functools
import glob
import random
import multiprocessing
import numpy as np
import tensorflow as tf
from featurizers.SpeechFeaturizer import read_raw_audio, speech_feature_extraction, compute_feature_dim

AUTOTUNE = tf.data.experimental.AUTOTUNE
TFRECORD_SHARDS = 16


def _float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def to_tfrecord(audio, transcript):
    feature = {
        "audio": _bytestring_feature([audio]),
        "transcript": _bytestring_feature([transcript])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


class Dataset:
    def __init__(self, data_path, mode="train"):
        self.data_path = data_path
        self.mode = mode
        self.samples = 0

    def __call__(self, speech_conf, text_featurizer, batch_size=32, augmentations=(),
                 sortagrad=False, builtin=False, fext=True, tfrecords_dir=None):
        if tfrecords_dir:
            if not os.path.exists(tfrecords_dir):
                os.makedirs(tfrecords_dir)
            self.create_tfrecords(tfrecords_dir, sortagrad)

        if self.mode == "train":
            if tfrecords_dir:
                return self.get_dataset_from_tfrecords(text_featurizer=text_featurizer,
                                                       augmentations=augmentations, speech_conf=speech_conf,
                                                       batch_size=batch_size, sort=sortagrad,
                                                       shuffle=True, builtin=builtin)
            return self.get_dataset_from_generator(text_featurizer=text_featurizer,
                                                   augmentations=augmentations, speech_conf=speech_conf,
                                                   batch_size=batch_size, shuffle=True,
                                                   sort=sortagrad, builtin=builtin)
        elif self.mode in ["eval", "test"]:
            if tfrecords_dir:
                if fext:
                    return self.get_dataset_from_tfrecords(text_featurizer=text_featurizer,
                                                           augmentations=[], speech_conf=speech_conf,
                                                           batch_size=batch_size, sort=sortagrad,
                                                           shuffle=True, builtin=builtin)
                return self.get_dataset_from_tfrecords_no_fext(text_featurizer=text_featurizer,
                                                               augmentations=[], speech_conf=speech_conf,
                                                               batch_size=batch_size, sort=sortagrad,
                                                               shuffle=False, builtin=builtin)
            return self.get_dataset_from_generator(text_featurizer=text_featurizer,
                                                   augmentations=[], speech_conf=speech_conf,
                                                   batch_size=batch_size, shuffle=False,
                                                   sort=sortagrad, builtin=builtin)
        else:
            raise ValueError(f"Mode must be either 'train', 'eval' or 'test': {self.mode}")

    @staticmethod
    def write_tfrecord_file(splitted_entries):
        shard_path, entries = splitted_entries
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for audio_file, _, transcript in entries:
                with open(audio_file, "rb") as f:
                    audio = f.read()
                example = to_tfrecord(audio, bytes(transcript, "utf-8"))
                out.write(example.SerializeToString())
                sys.stdout.write("\033[K")
                print(f"\rProcessed: {audio_file}", end="")
        print(f"\nCreated {shard_path}")

    def create_tfrecords(self, tfrecords_dir, sortagrad=False):
        print(f"Creating {self.mode}.tfrecord ...")
        self.tfrecords_dir = tfrecords_dir
        if not os.path.exists(self.tfrecords_dir):
            os.makedirs(self.tfrecords_dir)
        if glob.glob(os.path.join(tfrecords_dir, f"{self.mode}*.tfrecord")):
            return
        entries = self.create_entries(sortagrad)

        def get_shard_path(shard_id):
            return os.path.join(tfrecords_dir, f"{self.mode}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, TFRECORD_SHARDS + 1)]

        splitted_entries = np.array_split(entries, TFRECORD_SHARDS)
        with multiprocessing.Pool(TFRECORD_SHARDS) as pool:
            pool.map(self.write_tfrecord_file, zip(shards, splitted_entries))

    def create_entries(self, sort=False):  # Sort on entries, shuffle on dataset creation
        lines = []
        for file_path in self.data_path:
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of csv file
                lines += temp_lines[1:]
        # The files is "\t" seperated
        lines = [line.split("\t", 2) for line in lines]
        if sort:
            lines.sort(key=lambda item: float(item[1]))
        else:
            random.shuffle(lines)
        lines = np.array(lines)
        self.samples = len(lines)
        return lines

    @staticmethod
    def preprocess(audio, transcript, speech_conf, text_featurizer, augments):
        signal = read_raw_audio(audio.numpy(), speech_conf["sample_rate"])

        for augment in augments:
            if not augment.is_post:
                signal = augment(signal=signal, sample_rate=speech_conf["sample_rate"])

        features = speech_feature_extraction(signal, speech_conf)

        for augment in augments:
            if augment.is_post:
                features = augment(features)

        label = text_featurizer.compute_label_features(transcript.numpy().decode("utf-8"))
        label_length = tf.cast(tf.shape(label)[0], tf.int32)
        features = tf.convert_to_tensor(features, tf.float32)
        input_length = tf.cast(tf.shape(features)[0], tf.int32)
        return features, input_length, label, label_length

    @staticmethod
    def preprocess_no_fext(audio, transcript, speech_conf, text_featurizer, augments):
        signal = read_raw_audio(audio.numpy(), speech_conf["sample_rate"])

        for augment in augments:
            if not augment.is_post:
                signal = augment(signal=signal, sample_rate=speech_conf["sample_rate"])

        label = text_featurizer.compute_label_features(transcript.numpy().decode("utf-8"))
        return signal, label

    def parse_from_tfrecord(self, record, speech_conf, text_featurizer, augments, builtin=False, fext=True):
        feature_description = {
            "audio": tf.io.FixedLenFeature([], tf.string),
            "transcript": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        if fext:
            features, input_length, label, label_length = tf.py_function(
                functools.partial(self.preprocess, text_featurizer=text_featurizer,
                                  speech_conf=speech_conf, augments=augments),
                inp=[example["audio"], example["transcript"]],
                Tout=(tf.float32, tf.int32, tf.int32, tf.int32))
            if builtin:
                return (features, input_length, label, label_length), -1
            return features, input_length, label, label_length
        else:
            signal, label = tf.py_function(
                functools.partial(self.preprocess_no_fext, text_featurizer=text_featurizer,
                                  speech_conf=speech_conf, augments=augments),
                inp=[example["audio"], example["transcript"]],
                Tout=(tf.float32, tf.int32))
            if builtin:
                return (signal, label), -1
            return signal, label

    def parse_from_generator(self, signal, transcript, speech_conf, text_featurizer, augments, builtin=False):
        features, input_length, label, label_length = tf.py_function(
            functools.partial(self.preprocess, text_featurizer=text_featurizer,
                              speech_conf=speech_conf, augments=augments),
            inp=[signal, transcript],
            Tout=(tf.float32, tf.int32, tf.int32, tf.int32)
        )
        if builtin:
            return (features, input_length, label, label_length), -1
        return features, input_length, label, label_length

    def get_dataset_from_tfrecords(self, text_featurizer, augmentations, speech_conf,
                                   batch_size, sort=False, shuffle=True, builtin=False):
        # GET Records dataset
        pattern = os.path.join(self.tfrecords_dir, f"{self.mode}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)

        # CREATE dataset with augmentations (or not due to random)
        def parse(record):
            return self.parse_from_tfrecord(record, speech_conf, text_featurizer,
                                            augments=augmentations, builtin=builtin)

        dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)

        # SHUFFLE unbatched dataset (shuffle the elements with each other) if not using sortagrad
        if shuffle and not sort:
            dataset = dataset.shuffle(batch_size)

        # PADDED BATCH the dataset
        feature_dim, channel_dim = compute_feature_dim(speech_conf)
        if builtin:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=((tf.TensorShape([None, feature_dim, channel_dim]), tf.TensorShape([]),
                                tf.TensorShape([None]), tf.TensorShape([])),
                               tf.TensorShape([])),
                padding_values=((0., 0, text_featurizer.num_classes - 1, 0), 0)
            )
        else:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=(tf.TensorShape([None, feature_dim, channel_dim]), tf.TensorShape([]),
                               tf.TensorShape([None]), tf.TensorShape([])),
                padding_values=(0., 0, text_featurizer.num_classes - 1, 0)
            )

        # SHUFFLE the BATCHED dataset (only shuffle the batches with each other) if using sortagrad
        if shuffle and sort:
            dataset = dataset.shuffle(batch_size)

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def get_dataset_from_tfrecords_no_fext(self, text_featurizer, augmentations, speech_conf,
                                           batch_size, sort=False, shuffle=True, builtin=False):
        pattern = os.path.join(self.tfrecords_dir, f"{self.mode}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)

        def parse(record):
            return self.parse_from_tfrecord(record, speech_conf, text_featurizer,
                                            augments=augmentations, builtin=builtin, fext=False)

        dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)

        if shuffle and not sort:
            dataset = dataset.shuffle(batch_size)

        if builtin:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=((tf.TensorShape([None]), tf.TensorShape([None])), tf.TensorShape([])),
                padding_values=((0., text_featurizer.num_classes - 1), 0)
            )
        else:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None])),
                padding_values=(0., text_featurizer.num_classes - 1)
            )

        if shuffle and sort:
            dataset = dataset.shuffle(batch_size)

        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def get_dataset_from_generator(self, text_featurizer, augmentations, speech_conf,
                                   batch_size, shuffle=True, sort=False, builtin=False):
        entries = self.create_entries(sort)

        def gen():
            for audio_path, _, transcript in entries:
                with open(audio_path, "rb") as f:
                    signal = f.read()
                yield signal, bytes(transcript, "utf-8")

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
        )

        option = tf.data.Options()
        option.experimental_deterministic = False
        dataset = dataset.with_options(option)

        def parse(signal, transcript):
            return self.parse_from_generator(signal, transcript, speech_conf=speech_conf,
                                             text_featurizer=text_featurizer, augments=augmentations, builtin=builtin)

        dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)

        feature_dim, channel_dim = compute_feature_dim(speech_conf)

        if shuffle and not sort:
            dataset = dataset.shuffle(batch_size)

        if builtin:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=((tf.TensorShape([None, feature_dim, channel_dim]), tf.TensorShape([]),
                                tf.TensorShape([None]), tf.TensorShape([])), tf.TensorShape([])),
                padding_values=((0., 0, text_featurizer.num_classes - 1, 0), 0)
            )
        else:
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=(tf.TensorShape([None, feature_dim, channel_dim]), tf.TensorShape([]),
                               tf.TensorShape([None]), tf.TensorShape([])),
                padding_values=(0., 0, text_featurizer.num_classes - 1, 0)
            )

        if shuffle and sort:
            dataset = dataset.shuffle(batch_size)

        dataset = dataset.prefetch(AUTOTUNE)
        return dataset
