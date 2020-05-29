from __future__ import absolute_import

import tensorflow as tf


@tf.function
def shape_list(x):
    """Return list of dims, statically where possible."""
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
    return ret


@tf.function
def split_heads_2d(inputs, Nh):
    """Split channels into multiple heads."""
    B, H, W, d = shape_list(inputs)
    ret_shape = [B, H, W, Nh, d // Nh]
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 3, 1, 2, 4])


@tf.function
def combine_heads_2d(inputs):
    """Combine heads (inverse of split heads 2d)."""
    transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
    return tf.reshape(transposed, ret_shape)


@tf.function
def rel_to_abs(x):
    """Converts tensor from relative to aboslute indexing."""
    # [B, Nh, L, 2L−1]
    B, Nh, L, _ = shape_list(x)
    # Pad to shift from relative to absolute indexing.
    col_pad = tf.zeros((B, Nh, L, 1))
    x = tf.concat([x, col_pad], axis=3)
    flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
    flat_pad = tf.zeros((B, Nh, L - 1))
    flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
    # Reshape and slice out the padded elements.
    final_x = tf.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
    final_x = final_x[:, :, :L, L - 1:]
    return final_x


@tf.function
def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
    """Compute relative logits along one dimenion."""
    rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
    # Collapse height and heads
    rel_logits = tf.reshape(rel_logits, [-1, Nh * H, W, 2 * W - 1])
    rel_logits = rel_to_abs(rel_logits)
    # Shape it and tile height times
    rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
    rel_logits = tf.expand_dims(rel_logits, axis=3)
    rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
    # Reshape for adding to the logits.
    rel_logits = tf.transpose(rel_logits, transpose_mask)
    rel_logits = tf.reshape(rel_logits, [-1, Nh, H * W, H * W])
    return rel_logits


class relative_logits(tf.Module):
    def __init__(self, H, W, Nh, dkh, name="relative_logits"):
        super(relative_logits, self).__init__(name=name)
        self.H = H; self.W = W; self.Nh = Nh; self.dkh = dkh
        # Relative logits in width dimension first.
        self.rel_embeddings_w = tf.Variable(name='r_width',
                                            initial_value=tf.random.normal(shape=(2 * W - 1, dkh),
                                                                           mean=dkh**-0.5))
        # Relative logits in height dimension next.
        self.rel_embeddings_h = tf.Variable(name='r_height',
                                            initial_value=tf.random.normal(shape=(2 * H - 1, dkh),
                                                                           mean=dkh**-0.5))

    def __call__(self, q):
        """Compute relative logits."""
        # [B, Nh, HW, HW]
        rel_logits_w = relative_logits_1d(q, self.rel_embeddings_w, self.H, self.W, self.Nh, [0, 1, 2, 4, 3, 5])
        # For ease, we 1) transpose height and width,
        # 2) repeat the above steps and
        # 3) transpose to eventually put the logits
        # in their right positions.
        # [B, Nh, HW, HW]
        rel_logits_h = relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),
                                          self.rel_embeddings_h, self.W, self.H, self.Nh, [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w
