import tensorflow as tf

def add(a, b):
  return a + b

def conv2d(x, kernel_shape, n_out_channel, stride=(1, 1), pad="SAME", fmt="NHWC"):
  assert fmt in ("NHWC",)
  return tf.keras.layers.Conv2D(n_out_channel, kernel_shape, stride, padding=pad, data_format="channels_last", kernel_initializer=tf.compat.v1.initializers.he_normal(seed=0x12345678), bias_initializer=tf.compat.v1.constant_initializer(0.0)).apply(x)

def adv_pool(x, kernel_shape, stride, win_fn, trainable, pad="SAME", fmt="NHWC"):
  assert fmt in ("NHWC",)
  assert win_fn == ("boxcar", "boxcar")
  assert trainable == False
  return tf.keras.layers.AvgPool2D(kernel_shape, stride, padding=pad, data_format="channels_last").apply(x)

def resize_nearest(x, size, fmt="NHWC"):
  assert fmt == "NHWC"
  return tf.image.resize(x, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def get_shape(x):
  return tf.shape(input=x)

def ret_new(x):
  return x

def relu(x):
  return tf.nn.relu(x)

def lrelu(x):
  return tf.nn.leaky_relu(x, alpha=0.01)

def sigmoid(x):
  return tf.sigmoid(x)

def concat(l, axis):
  return tf.concat(l, axis=axis)

def count_parameter():
  total_parameters = 0
  for variable in tf.compat.v1.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
      variable_parameters *= int(dim)
    total_parameters += variable_parameters
  return total_parameters