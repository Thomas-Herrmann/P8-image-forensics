import tensorflow as tf


A = tf.constant(
    [[1,1,1,0],
      [1,8,1,0],
      [1,1,9,0],
      [0,0,0,3]]
)

Nh = tf.constant(4)

s = (A.shape)[:-1]

print(f"{s + [Nh, tf.constant(s) // Nh]}")