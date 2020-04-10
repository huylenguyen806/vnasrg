from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import ctc_loss_func
from utils.Schedules import BoundExponentialDecay


def decode_lambda_func(args, **arguments):
  y_pred, input_length = args
  decoder = arguments["decoder"]
  result = decoder.decode(probs=y_pred,
                          input_length=tf.squeeze(input_length,
                                                  axis=-1))
  return result


def test_lambda_func(args, **arguments):
  y_pred, input_length, labels = args
  decoder = arguments["decoder"]
  predictions = decoder.decode(probs=y_pred,
                               input_length=tf.squeeze(input_length,
                                                       axis=-1))
  string_labels = decoder.convert_to_string(labels)
  predictions = tf.expand_dims(predictions, 1)
  string_labels = tf.expand_dims(string_labels, 1)
  outputs = tf.concat([predictions, string_labels], axis=-1)

  def cal_each_er(elem):
    pred = elem[0].numpy().decode("utf-8")
    target = elem[1].numpy().decode("utf-8")
    print("Prediction: ", pred)
    print("Groundtruth: ", target)
    cal_wer, wer_count = wer(decode=pred, target=target)
    cal_cer, cer_count = cer(decode=pred, target=target)
    return tf.convert_to_tensor([cal_wer, wer_count, cal_cer, cer_count])

  return tf.map_fn(cal_each_er, outputs, dtype=tf.int32)


def create_ctc_model(num_classes, num_feature_bins,
                     learning_rate, base_model,
                     min_lr=0.0, streaming_size=None):
  if streaming_size:
    # Fixed input shape is required for live streaming audio
    features = tf.keras.layers.Input(
      batch_shape=(1, streaming_size, num_feature_bins, 1),
      dtype=tf.float32,
      name="features")
    outputs = base_model(features=features, streaming=True)
  else:
    features = tf.keras.layers.Input(
      shape=(None, num_feature_bins, 1),
      dtype=tf.float32,
      name="features")
    outputs = base_model(features=features, streaming=False)

  batch_size = tf.shape(outputs)[0]
  n_hidden = outputs.get_shape().as_list()[-1]
  # reshape from [B, T, A] --> [B*T, A].
  # Output shape: [n_steps * batch_size, n_hidden]
  outputs = tf.reshape(outputs, [-1, n_hidden])

  # Fully connected layer
  outputs = tf.keras.layers.Dense(units=num_classes,
                                  activation='softmax',
                                  name="fully_connected",
                                  use_bias=True)(outputs)

  outputs = tf.reshape(outputs,
                       [batch_size, -1, num_classes],
                       name="logits")

  model = tf.keras.Model(inputs=features, outputs=outputs)

  # y_true is None because of dummy label and loss is calculated
  # in the layer lambda
  model.compile(
    optimizer=base_model.optimizer(
      learning_rate=BoundExponentialDecay(
        min_lr=min_lr,
        initial_learning_rate=learning_rate,
        decay_steps=5000,
        decay_rate=0.9,
        staircase=True)),
    loss=ctc_loss_func
  )
  return model
