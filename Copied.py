import tensorflow as tf
import tensorflow.compat.v1 as tfc

def shape_list(x):
    """Return list of dims, statically where possible."""
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
    return ret


def split_heads_2d(inputs, Nh):
    """Split channels into multiple heads."""
    B, H, W, d = shape_list(inputs)
    ret_shape = [B, H, W, Nh, d // Nh]
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 3, 1, 2, 4])


def combine_heads_2d(inputs):
    """Combine heads (inverse of split heads 2d)."""
    transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
    return tf.reshape(transposed, ret_shape)


def rel_to_abs(x):
    """Converts tensor from relative to aboslute indexing."""
    # [B, Nh, L, 2L-1]
    B, Nh, L, = shape_list(x)
    # Pad to shift from relative to absolute indexing.
    col_pad = tf.zeros((B, Nh, L, 1))
    x = tf.concat([x, col_pad], axis=3)
    flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
    flat_pad = tf.zeros((B, Nh, L-1))
    flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
    # Reshape and slice out the padded elements.
    final_x = tf.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
    final_x = final_x[:, :, :L, L-1:]
    return final_x


def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
    """Compute relative logits along one dimenion."""
    rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
    # Collapse height and heads
    rel_logits = tf.reshape(
    rel_logits, [-1, Nh * H, W, 2 * W-1])
    rel_logits = rel_to_abs(rel_logits)
    # Shape it and tile height times
    rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
    rel_logits = tf.expand_dims(rel_logits, axis=3)
    rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
    # Reshape for adding to the logits.
    rel_logits = tf.transpose(rel_logits, transpose_mask)
    rel_logits = tf.reshape(rel_logits, [-1, Nh, H*W, H*W])
    return rel_logits


def relative_logits(q, H, W, Nh, dkh):
    """Compute relative logits."""
    # Relative logits in width dimension first.
    rel_embeddings_w = tf.get_variable('r_width', shape=(2*W - 1, dkh),
    initializer=tf.random_normal_initializer(dkh**-0.5))
    # [B, Nh, HW, HW]
    rel_logits_w = relative_logits_1d(q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5])
    # Relative logits in height dimension next.
    # For ease, we 1) transpose height and width,
    # 2) repeat the above steps and
    # 3) transpose to eventually put the logits
    # in their right positions.
    rel_embeddings_h = tf.get_variable('r_height', shape=(2 * H - 1, dkh), initializer=tf.random_normal_initializer(dkh**-0.5))
    # [B, Nh, HW, HW]
    rel_logits_h = relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]), rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
    return rel_logits_h, rel_logits_w


def self_attention_2d(inputs, dk, dv, Nh, relative=True):
    """2d relative self-attention."""
    _, H, W,_ = shape_list(inputs)
    dkh = dk // Nh
    dvh = dv // Nh
    flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])
    # Compute q, k, v
    kqv = tfc.layers.conv2d(inputs, 2 * dk + dv, 1)
    k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
    q *= dkh ** -0.5 # scaled dot-product
    # After splitting, shape is [B, Nh, H, W, dkh or dvh]
    q = split_heads_2d(q, Nh)
    k = split_heads_2d(k, Nh)
    v = split_heads_2d(v, Nh)
    # [B, Nh, HW, HW]
    logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh), transpose_b=True)
    if relative:
        rel_logits_h, rel_logits_w = relative_logits(q, H, W, Nh, dkh)
        logits += rel_logits_h
        logits += rel_logits_w

    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flatten_hw(v, dvh))
    attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
    attn_out = combine_heads_2d(attn_out)
    # Project heads
    attn_out = tfc.layers.conv2d(attn_out, dv, 1)
    return attn_out


def augmented_conv2d(X, Fout, k, dk, dv, Nh, relative):
    conv_out = tfc.layers.conv2d(inputs=X, filters=Fout - dv, kernel_size=k, padding='same')
    attn_out = self_attention_2d(X, dk, dv, Nh, relative=relative)
    return tf.concat([conv_out, attn_out], axis=3)

