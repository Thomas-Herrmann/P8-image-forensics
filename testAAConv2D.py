import tensorflow as tf
import AAConv2D as AA
import Copied as Orig
import numpy as np


is_rel = False
Fout   = 3
k      = 3 
dk     = 2
dv     = 2
Nh     = 2


X = tf.constant(np.ones((1, 8, 8, 1)))


tf.random.set_seed(54)

aaObj    = AA.AAConv2D(Fout, k, dk, dv, Nh, is_rel)
aa_out   = aaObj(X)

tf.random.set_seed(54)

orig_out = Orig.augmented_conv2d(X, Fout, k, dk, dv, Nh, is_rel)


print(f"AA:\n{aa_out}\n\nOG:\n{orig_out}")