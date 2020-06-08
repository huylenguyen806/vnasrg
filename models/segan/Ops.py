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
from __future__ import absolute_import

import tensorflow as tf


class DownConv(tf.keras.layers.Layer):
    def __init__(self, depth, kwidth=5, pool=2, name="downconv", **kwargs):
        super(DownConv, self).__init__(name=name, **kwargs)
        self.layer = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=(kwidth, 1),
            strides=(pool, 1),
            padding="same",
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            bias_initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(DownConv, self).get_config()
        config.update({"layer": self.layer})
        return config

    def from_config(self, config):
        return self(**config)


class DeConv(tf.keras.layers.Layer):
    def __init__(self, depth, kwidth=5, dilation=2, name="deconv", **kwargs):
        super(DeConv, self).__init__(name=name, **kwargs)
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters=depth,
            kernel_size=(kwidth, 1),
            strides=(dilation, 1),
            padding="same",
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            bias_initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(DeConv, self).get_config()
        config.update({"layer": self.layer})
        return config

    def from_config(self, config):
        return self(**config)


class VirtualBatchNorm:
    def __init__(self, x, name, epsilon=1e-5):
        assert isinstance(epsilon, float)
        self.epsilon = epsilon
        self.name = name
        self.batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        self.gamma = tf.Variable(
            initial_value=tf.random_normal_initializer(1., 0.02)(
                shape=[x.get_shape().as_list()[-1]]),
            name="gamma", trainable=True
        )
        self.beta = tf.Variable(
            initial_value=tf.constant_initializer(0.)(
                shape=[x.get_shape().as_list()[-1]]),
            name="beta", trainable=True
        )
        mean, var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)
        self.mean = mean
        self.variance = var

    def __call__(self, x):
        new_coeff = 1. / (self.batch_size + 1.)
        old_coeff = 1. - new_coeff
        new_mean, new_var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)
        new_mean = new_coeff * new_mean + old_coeff * self.mean
        new_var = new_coeff * new_var + old_coeff * self.variance
        return tf.nn.batch_normalization(x, mean=new_mean, variance=new_var,
                                         offset=self.beta, scale=self.gamma,
                                         variance_epsilon=self.epsilon)


class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, name, noise_std, **kwargs):
        super(GaussianNoise, self).__init__(trainable=False, name=name, **kwargs)
        self.noise_std = noise_std

    def call(self, inputs, training=False):
        noise = tf.keras.backend.random_normal(shape=tf.shape(inputs),
                                               mean=0.0, stddev=self.noise_std,
                                               dtype=tf.float32)
        return inputs + noise


class Reshape1to3(tf.keras.layers.Layer):
    def __init__(self, name="reshape_1_to_3", **kwargs):
        super(Reshape1to3, self).__init__(trainable=False, name=name, **kwargs)

    def call(self, inputs, training=False):
        width = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [-1, width, 1, 1])

    def get_config(self):
        config = super(Reshape1to3, self).get_config()
        return config

    def from_config(self, config):
        return self(**config)


class Reshape3to1(tf.keras.layers.Layer):
    def __init__(self, name="reshape_3_to_1", **kwargs):
        super(Reshape3to1, self).__init__(trainable=False, name=name, **kwargs)

    def call(self, inputs, training=False):
        width = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [-1, width])

    def get_config(self):
        config = super(Reshape3to1, self).get_config()
        return config

    def from_config(self, config):
        return self(**config)


class SeganPrelu(tf.keras.layers.Layer):
    def __init__(self, name="segan_prelu", **kwargs):
        super(SeganPrelu, self).__init__(trainable=True, name=name, **kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha",
                                     shape=input_shape[-1],
                                     initializer=tf.keras.initializers.zeros,
                                     dtype=tf.float32,
                                     trainable=True)

    def call(self, x, training=False):
        pos = tf.nn.relu(x)
        neg = self.alpha * (x - tf.abs(x)) * .5
        return pos + neg

    def get_config(self):
        config = super(SeganPrelu, self).get_config()
        return config

    def from_config(self, config):
        return self(**config)
