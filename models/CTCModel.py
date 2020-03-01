from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import wer, cer


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name="CTC Loss")

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.squeeze(y_pred))


def ctc_lambda_func(args):
    y_pred, input_length, labels, label_length = args
    label_length = tf.squeeze(label_length)
    input_length = tf.squeeze(input_length)
    # sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length)
    # y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) +
    #                      tf.keras.backend.epsilon())
    # return tf.expand_dims(
    #     tf.compat.v1.nn.ctc_loss(labels=sparse_labels,
    #                              inputs=y_pred,
    #                              sequence_length=input_length,
    #                              ignore_longer_outputs_than_inputs=True), 1)
    return tf.expand_dims(
        tf.nn.ctc_loss(labels=labels, logits=y_pred, label_length=label_length,
                       logit_length=input_length, logits_time_major=False), 1)


def decode_lambda_func(args, **arguments):
    y_pred, input_length = args
    decoder = arguments["decoder"]
    return decoder.decode(probs=y_pred, input_length=tf.squeeze(input_length))


def test_lambda_func(args, **arguments):
    y_pred, input_length, labels = args
    decoder = arguments["decoder"]
    predictions = decoder.decode(
        probs=y_pred, input_length=tf.squeeze(input_length))
    string_labels = decoder.convert_to_string(labels)
    outputs = tf.concat([predictions, string_labels], axis=0)

    def cal_each_er(elem):
        cal_wer = wer(decode=elem[0], target=elem[1])
        cal_cer = cer(decode=elem[0], target=elem[1])
        return tf.convert_to_tensor([cal_wer, cal_cer])

    return tf.map_fn(cal_each_er, outputs, dtype=tf.float64)


def create_ctc_model(num_classes, num_feature_bins, learning_rate,
                     base_model, decoder, mode="train"):
    # Convolution layers
    features = tf.keras.layers.Input(shape=(None, num_feature_bins, 1),
                                     dtype=tf.float32, name="features")
    input_length = tf.keras.layers.Input(shape=(1,), dtype=tf.int32,
                                         name="input_length")
    labels = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                   name="labels")
    label_length = tf.keras.layers.Input(shape=(1,), dtype=tf.int32,
                                         name="label_length")

    outputs = base_model(features=features)

    # Fully connected layer
    outputs = tf.keras.layers.Dense(units=num_classes,
                                    activation=tf.keras.activations.softmax,
                                    use_bias=True)(outputs)

    if mode == "train":
        # Lambda layer for computing loss function
        loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,),
                                          name="ctc_loss")([outputs, input_length,
                                                            labels, label_length])
        train_model = tf.keras.Model(inputs={
            "features": features,
            "input_length": input_length,
            "labels": labels,
            "label_length": label_length
        }, outputs=loss_out)

        # y_true is None because of dummy label and loss is calculated in the layer lambda
        train_model.compile(
            optimizer=base_model.optimizer(lr=learning_rate),
            loss=CTCLoss()
        )
        return train_model
    elif mode == "infer":
        # Lambda layer for decoding to text
        decode_out = tf.keras.layers.Lambda(decode_lambda_func, output_shape=(None,),
                                            name="ctc_decoder",
                                            arguments={"decoder": decoder},
                                            dynamic=True)([outputs, input_length])
        infer_model = tf.keras.Model(inputs={
            "features": features,
            "input_length": input_length
        }, outputs=decode_out)

        infer_model.compile(
            optimizer=base_model.optimizer(lr=learning_rate),
            loss={"ctc_decoder": lambda y_true, y_pred: y_pred}
        )
        return infer_model
    elif mode == "test":
        # Lambda layer for analysis
        test_out = tf.keras.layers.Lambda(test_lambda_func, output_shape=(None,),
                                          name="ctc_test",
                                          arguments={"decoder": decoder},
                                          dynamic=True)([outputs, input_length,
                                                         labels])
        test_model = tf.keras.Model(inputs={
            "features": features,
            "input_length": input_length,
            "labels": labels
        }, outputs=test_out)

        test_model.compile(
            optimizer=base_model.optimizer(lr=learning_rate),
            loss={"ctc_test": lambda y_true, y_pred: y_pred}
        )
        return test_model
    else:
        raise ValueError("mode must be either 'train', 'infer' or 'test'")
