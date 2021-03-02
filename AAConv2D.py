import tensorflow as tf
from tensorflow.nn import softmax


class AAConv2D(tf.keras.layers.Conv2D):

    def __init__(self, Fout, k, dk, dv, Nh, is_rel):
        
        super(AAConv2D, self).__init__(filters = Fout - dv, kernel_size = k, padding = "same")

        self.dk     = dk
        self.dkh    = dk // Nh
        self.dv     = dv
        self.dvh    = dv // Nh
        self.Nh     = Nh
        self.is_rel = is_rel
        self.kqv    = tf.keras.layers.Conv2D(filters = self.dk + self.dk + self.dv, kernel_size = 1, padding = "same")
        self.satt   = tf.keras.layers.Conv2D(filters = self.dv, kernel_size = 1, padding = "same")


    def build(self, input_shape):
        
        super(AAConv2D, self).build(input_shape)
        
        self.kqv.build(input_shape)
        self.satt.build(input_shape)
        
        initializer = tf.random_normal_initializer(self.dkh ** -0.5)
        H, W        = input_shape[1:-1]

        self.r_height = tf.Variable(initializer(shape = [2 * H - 1, self.dkh], dtype = tf.float32))
        self.r_width  = tf.Variable(initializer(shape = [2 * W - 1, self.dkh], dtype = tf.float32))


    @tf.autograph.experimental.do_not_convert
    def call(self, X, training=None):
        
        conv2D_out = super(AAConv2D, self).call(X)
        satt_out   = self.__mhead_satt2D(X)

        return tf.concat([conv2D_out, satt_out], axis = 3)

    
    def __mhead_satt2D(self, X):
        
        k, q, v = tf.split(self.kqv.call(X), num_or_size_splits = [self.dk, self.dk, self.dv], axis = 3)
        q      *= self.dkh ** -0.5
        k, q, v = [self.__split_heads2D(T) for T in [k, q, v]]

        H, W       = self.__tensor_dims(X)[1:-1]
        flatten_hw = lambda T, dh: tf.reshape(T, [-1, self.Nh, H * W, dh])
        logits     = tf.matmul(flatten_hw(q, self.dkh), flatten_hw(k, self.dkh), transpose_b = True)

        if self.is_rel:
            rel_logitsW, rel_logitsH = self.__rel_logits(q, H, W)

            logits += rel_logitsW + rel_logitsH
        
        weights = softmax(logits)

        weighted       = tf.matmul(weights, flatten_hw(v, self.dvh))
        deflattened    = tf.reshape(weighted, [-1, self.Nh, H, W, self.dvh])
        heads_combined = self.__combine_heads2D(deflattened)

        return self.satt.call(heads_combined)


    def __tensor_dims(self, X):

        static  = X.shape.as_list()
        dynamic = tf.shape(X)

        return [static[i] or dynamic[i] for i in range(len(static))]


    def __split_heads2D(self, X):

        B, H, W, d  = self.__tensor_dims(X)
        depth_split = tf.reshape(X, [B, H, W, self.Nh, d // self.Nh])

        # [B, H, W, Nh, dh] --> [B, Nh, H, W, dh]
        return tf.transpose(depth_split, [0, 3, 1, 2, 4]) 


    def __combine_heads2D(self, X):

        # [B, Nh, H, W, dh] --> [B, H, W, Nh, dh]
        detransposed    = tf.transpose(X, [0, 2, 3, 1, 4])
        B, H, W, Nh, dh = self.__tensor_dims(detransposed)

        return tf.reshape(detransposed, [B, H, W, Nh * dh])


    def __rel_logits(self, q, H, W):

        rel_logitsW = self.__rel_logits1D(q, self.r_width, H, W, [0, 1, 2, 4, 3, 5])
        rel_logitsH = self.__rel_logits1D(tf.transpose(q, [0, 1, 3, 2, 4]), self.r_height, W, H, [0, 1, 4, 2, 5, 3])

        return rel_logitsW, rel_logitsH
                                                

    def __rel_logits1D(self, q, embedded, H, W, transp_mask):

        rel_logits      = tf.einsum("b h x y d, m d -> b h x y m", q, embedded)
        heads_collapsed = tf.reshape(rel_logits, [-1, self.Nh * H, W, 2 * W - 1])
        abs_pos_logits  = self.__rel_pos_to_abs(heads_collapsed)

        reshaped_abs_pos_logits = tf.reshape(abs_pos_logits, [-1, self.Nh, H, W, W])
        H_1s_dim_inserted       = tf.tile(tf.expand_dims(reshaped_abs_pos_logits, axis = 3), [1, 1, 1, H, 1, 1])
        
        return tf.reshape(tf.transpose(H_1s_dim_inserted, transp_mask), [-1, self.Nh, H * W, W * H])


    def __rel_pos_to_abs(self, T):

        B, rNh, L, _     = self.__tensor_dims(T)
        padded           = tf.concat([T, tf.zeros([B, rNh, L, 1])], axis = 3)
        collapsed        = tf.reshape(padded, [B, rNh, L * 2 * L])
        collapsed_padded = tf.concat([collapsed, tf.zeros([B, rNh, L - 1])], axis = 2)
        reshaped         = tf.reshape(collapsed_padded, [B, rNh, L + 1, 2 * L - 1])
        
        return tf.slice(reshaped, [0, 0, 0, L - 1], [B, rNh, L, L])