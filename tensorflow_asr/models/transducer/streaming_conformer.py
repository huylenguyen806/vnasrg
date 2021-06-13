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
""" http://arxiv.org/abs/1811.06621 """

import tensorflow as tf

from ..layers.subsampling import TimeReduction
# from .transducer import Transducer
from ...utils import data_util, math_util
# from ...utils.utils import get_rnn, merge_two_last_dims, shape_list
from .conformer import Conformer

L2 = tf.keras.regularizers.l2(1e-6)

class StreamingConformer(Conformer):
    """
    Attempt at implementing Streaming Conformer Transducer. (see: https://arxiv.org/pdf/2010.11395.pdf).

    Three main differences:
    - Inputs are splits into chunks.
    - Masking is used for MHSA to select the chunks to be used at each timestep. (Allows for parallel training.)
    - Added parameter `streaming` to ConformerEncoder, ConformerBlock and ConvModule. Inside ConvModule, the layer DepthwiseConv2D has padding changed to "causal" when `streaming==True`.

    NOTE: Masking is applied just as regular masking along with the inputs.
    """
    def __init__(self,
                 vocabulary_size: int,
                 encoder_subsampling: dict,
                 encoder_positional_encoding: str = "sinusoid",
                 encoder_dmodel: int = 144,
                 encoder_num_blocks: int = 16,
                 encoder_head_size: int = 36,
                 encoder_num_heads: int = 4,
                 encoder_mha_type: str = "relmha",
                 encoder_kernel_size: int = 32,
                 encoder_depth_multiplier: int = 1,
                 encoder_fc_factor: float = 0.5,
                 encoder_dropout: float = 0,
                 encoder_trainable: bool = True,
                 prediction_embed_dim: int = 512,
                 prediction_embed_dropout: int = 0,
                 prediction_num_rnns: int = 1,
                 prediction_rnn_units: int = 320,
                 prediction_rnn_type: str = "lstm",
                 prediction_rnn_implementation: int = 2,
                 prediction_layer_norm: bool = True,
                 prediction_projection_units: int = 0,
                 prediction_trainable: bool = True,
                 joint_dim: int = 1024,
                 joint_activation: str = "tanh",
                 prejoint_linear: bool = True,
                 postjoint_linear: bool = False,
                 joint_mode: str = "add",
                 joint_trainable: bool = True,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name: str = "streaming_conformer",
                 **kwargs):

        self.streaming = True # Hardcoded value. Initializes Conformer with `streaming = True`.
        super(StreamingConformer, self).__init__(
            vocabulary_size=vocabulary_size,
            encoder_subsampling=encoder_subsampling,
            encoder_positional_encoding=encoder_positional_encoding,
            encoder_dmodel=encoder_dmodel,
            encoder_num_blocks=encoder_num_blocks,
            encoder_head_size=encoder_head_size,
            encoder_num_heads=encoder_num_heads,
            encoder_mha_type=encoder_mha_type,
            encoder_depth_multiplier=encoder_depth_multiplier,
            encoder_kernel_size=encoder_kernel_size,
            encoder_fc_factor=encoder_fc_factor,
            encoder_dropout=encoder_dropout,
            encoder_trainable=encoder_trainable,
            prediction_embed_dim=prediction_embed_dim,
            prediction_embed_dropout=prediction_embed_dropout,
            prediction_num_rnns=prediction_num_rnns,
            prediction_rnn_units=prediction_rnn_units,
            prediction_rnn_type=prediction_rnn_type,
            prediction_rnn_implementation=prediction_rnn_implementation,
            prediction_layer_norm=prediction_layer_norm,
            prediction_projection_units=prediction_projection_units,
            prediction_trainable=prediction_trainable,
            joint_dim=joint_dim,
            joint_activation=joint_activation,
            prejoint_linear=prejoint_linear,
            postjoint_linear=postjoint_linear,
            joint_mode=joint_mode,
            joint_trainable=joint_trainable,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            streaming=self.streaming,
            name=name,
            **kwargs
        )
        self.dmodel = encoder_dmodel
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor

    def make(self, input_shape, prediction_shape=[None], batch_size=None):
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        mask = tf.keras.Input(shape=[None, None], batch_size=batch_size, dtype=tf.int32)
        self(
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length,
                predictions=predictions,
                predictions_length=predictions_length,
                mask=mask
            ),
            training=False
        )

    def call(self, inputs, training=False, **kwargs):
        enc = self.encoder(inputs["inputs"], training=training, mask=inputs["mask"], **kwargs)
        pred = self.predict_net([inputs["predictions"], inputs["predictions_length"]], training=training, **kwargs)
        logits = self.joint_net([enc, pred], training=training, **kwargs)
        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(inputs["inputs_length"], self.time_reduction_factor)
        )
