import geneop
from typing import List

import tensorflow as tf

gene_len = 1

unify_counter = [0]
def count() -> int:
  unify_counter[0] += 1
  return unify_counter[0]

def conv2d(x, kernel_shape, n_out_channel, stride=(1, 1), pad="SAME", fmt="NHWC", scope=None, reuse=False):
  assert fmt in ("NCHW", "NHWC")
  if fmt == "NCHW":
    _, c, _, _ = x.get_shape()
    bias_shape = (1, n_out_channel, 1, 1)
  else:
    _, _, _, c = x.get_shape()
    bias_shape = (1, 1, 1, n_out_channel)
  kernel_shape = (*kernel_shape, c, n_out_channel)
  stride = (1, *stride, 1)
  with tf.compat.v1.variable_scope(scope, default_name="%d_conv2d" % (count(),), reuse=reuse):
    weight = tf.compat.v1.get_variable("weight", kernel_shape, initializer=tf.compat.v1.initializers.he_normal(seed=None), trainable=True)
    bias = tf.compat.v1.get_variable("bias", bias_shape, initializer=tf.compat.v1.constant_initializer(0.0), trainable=True)
    return tf.nn.conv2d(input=x, filters=weight, strides=stride, padding=pad, data_format=fmt) + bias

def relu(x):
  return tf.nn.relu(x)

def concat(l, axis):
  return tf.concat(l, axis=axis)

def build_from_gene(x, n_out_ch: int, seq: List[bool]):
  assert seq == [False,]
  x = tf.transpose(x, (0, 2, 1, 3))
  v = x
  a = conv2d(v, (21, 13), 12)
  b = conv2d(v, (9, 7), 3)
  c = conv2d(v, (3, 3), 3)
  v = relu(concat([a, b, c], axis=-1))

  a = conv2d(v, (21, 13), 3)
  b = conv2d(v, (9, 7), 16)
  c = conv2d(v, (3, 3), 3)
  v = relu(concat([a, b, c], axis=-1))

  a = conv2d(v, (21, 13), 3)
  b = conv2d(v, (9, 7), 12)
  c = conv2d(v, (3, 3), 7)
  v = relu(concat([a, b, c], axis=-1))

  a = conv2d(v, (21, 13), 3)
  b = conv2d(v, (9, 7), 3)
  c = conv2d(v, (3, 3), 32)
  v = relu(concat([a, b, c], axis=-1))

  a = conv2d(v, (21, 13), 3)
  b = conv2d(v, (9, 7), 12)
  c = conv2d(v, (3, 3), 7)
  v = relu(concat([a, b, c], axis=-1))

  a = conv2d(v, (21, 13), 3)
  b = conv2d(v, (9, 7), 16)
  c = conv2d(v, (3, 3), 3)
  v = relu(concat([a, b, c], axis=-1))

  a = conv2d(v, (21, 13), 12)
  b = conv2d(v, (9, 7), 3)
  c = conv2d(v, (3, 3), 3)
  v = relu(concat([a, b, c], axis=-1))

  return tf.transpose(conv2d(v, (512, 15), n_out_ch), (0, 2, 1, 3))