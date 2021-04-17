from typing import List, Dict, Tuple, Callable

def add(a, b):
  return ("add", a, b)

def conv2d(x, kernel_shape, n_out_channel, stride=(1, 1), pad="SAME", fmt="NHWC", scope=None, reuse=False):
  assert fmt in ("NCHW", "NHWC")
  return ("conv2d", x, tuple(kernel_shape), int(n_out_channel), tuple(stride), pad, fmt)

def sconv2d(x, kernel_shape, n_channel_mul, n_out_channel, stride=(1, 1), pad="SAME", fmt="NHWC", scope=None, reuse=False):
  assert fmt in ("NCHW", "NHWC")
  return ("conv2d", x, tuple(kernel_shape), int(n_channel_mul), int(n_out_channel), tuple(stride), pad, fmt)
  
def layer_norm(x, trainable=True, scope=None, reuse=None):
  return ("layer_norm", x, trainable)

def mean_norm(x, trainable=True, scope=None, reuse=None):
  return ("mean_norm", x, trainable)

def inst_norm(x, trainable=True, scope=None, reuse=None):
  return ("inst_norm", x, trainable)

def adv_pool(x, kernel_shape, stride, win_fn, trainable, pad="SAME", fmt="NHWC", scope=None, reuse=False):
  assert fmt in ("NCHW", "NHWC")
  return ("adv_pool", x, tuple(kernel_shape), tuple(stride), win_fn, trainable, pad, fmt)

def resize_nearest(x, size, fmt="NHWC"):
  assert fmt == "NHWC"
  return (x, size)

def get_shape(x):
  return (0, 1, 2, 3)

def ret_new(x):
  return hash(x)

def relu(x):
  return ("relu", x)

def lrelu(x):
  return ("lrelu", x)

def sigmoid(x):
  return ("sigmoid", x)

def concat(l, axis):
  return ("concat", tuple(l), axis)