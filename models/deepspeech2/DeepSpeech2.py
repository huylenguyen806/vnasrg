"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from models.deepspeech2.RowConv1D import RowConv1D
from models.deepspeech2.SequenceBatchNorm import SequenceBatchNorm


class DeepSpeech2:
  def __init__(self, conv_type=2, num_rnn=5, rnn_units=256, filters=(32, 32, 96),
               kernel_size=((11, 41), (11, 21), (11, 21)), strides=((2, 2), (1, 2), (1, 2)),
               optimizer=tf.keras.optimizers.SGD(lr=0.0002, momentum=0.99, nesterov=True),
               is_bidirectional=False, is_rowconv=False, pre_fc_units=1024):
    self.optimizer = optimizer
    self.num_rnn = num_rnn
    self.rnn_units = rnn_units
    self.filters = filters
    self.kernel_size = kernel_size
    self.is_bidirectional = is_bidirectional
    self.is_rowconv = is_rowconv
    self.pre_fc_units = pre_fc_units
    self.strides = strides
    self.conv_type = conv_type
    assert len(strides) == len(filters) == len(kernel_size)
    assert conv_type in [1, 2]

  @staticmethod
  def clipped_relu(x):
    return tf.keras.activations.relu(x, max_value=20)

  @staticmethod
  def merge_filter_to_channel(x):
    batch_size = tf.shape(x)[0]
    f, c = x.get_shape().as_list()[2:]
    return tf.reshape(x, [batch_size, -1, f * c])

  def __call__(self, features, streaming=False):
    layer = tf.keras.layers.BatchNormalization()(features)
    if self.conv_type == 2:
      layer = tf.expand_dims(layer, -1)
      conv = tf.keras.layers.Conv2D
    else:
      conv = tf.keras.layers.Conv1D
      ker_shape = np.shape(self.kernel_size)
      stride_shape = np.shape(self.strides)
      assert len(ker_shape) == 1 and len(stride_shape) == 1

    for i, fil in enumerate(self.filters):
      layer = conv(filters=fil, kernel_size=self.kernel_size[i],
                   strides=self.strides[i], padding="same",
                   activation=self.clipped_relu, name=f"cnn_{i}")(layer)

    layer = tf.keras.layers.BatchNormalization()(layer)

    if self.conv_type == 2:
      layer = self.merge_filter_to_channel(layer)

    # Convert to time_major only for bi_directional
    if self.is_bidirectional:
      layer = tf.transpose(layer, [1, 0, 2])

    # RNN layers
    for i in range(self.num_rnn):
      if self.is_bidirectional:
        layer = tf.keras.layers.Bidirectional(
          tf.keras.layers.GRU(units=self.rnn_units, dropout=0.2,
                              activation='tanh', recurrent_activation='sigmoid',
                              use_bias=True, recurrent_dropout=0.0,
                              return_sequences=True, unroll=False, implementation=2,
                              time_major=True, stateful=False, name=f"blstm_{i}"))(layer)
        layer = SequenceBatchNorm(time_major=True, name=f"sequence_wise_bn_{i}")(layer)
      else:
        layer = tf.keras.layers.GRU(units=self.rnn_units, dropout=0.2,
                                    activation='tanh', recurrent_activation='sigmoid',
                                    use_bias=True, recurrent_dropout=0.0,
                                    return_sequences=True, unroll=False, implementation=2,
                                    time_major=False, stateful=streaming, name=f"lstm_{i}")(layer)
        layer = SequenceBatchNorm(time_major=False, name=f"sequence_wise_bn_{i}")(layer)
        if self.is_rowconv:
          layer = RowConv1D(filters=self.rnn_units, future_context=2, name=f"row_conv_{i}")(layer)

    # Convert to batch_major
    if self.is_bidirectional:
      layer = tf.transpose(layer, [1, 0, 2])

    if self.pre_fc_units > 0:
      layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=self.pre_fc_units, activation=self.clipped_relu,
                              use_bias=True), name="hidden_fc")(layer)
      layer = tf.keras.layers.BatchNormalization()(layer)

    return layer
