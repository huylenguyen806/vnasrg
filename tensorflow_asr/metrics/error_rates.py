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


class ErrorRate(tf.keras.metrics.Metric):
    """Metric for WER or CER"""

    def __init__(self, func, name="error_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.numerator = self.add_weight(name="numerator", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")
        self.func = func

    def update_state(self, decode: tf.Tensor, target: tf.Tensor):
        n, d = self.func(decode, target)
        self.numerator.assign_add(n)
        self.denominator.assign_add(d)

    def result(self):
        return tf.math.divide(self.numerator, self.denominator)
