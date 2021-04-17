import geneop
from typing import List

import tensorflow as tf
from sa2_modules import att

gene_len = 3

def cnv(inp, kernel_shape, scope_name, stride=[1,1,1,1], dorelu=True,
  weight_init_fn=tf.compat.v1.random_normal_initializer,
  bias_init_fn=tf.compat.v1.constant_initializer, bias_init_val=0.0, pad='SAME',):

  with tf.compat.v1.variable_scope(scope_name):
    std = 1 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])
    std = std ** 0.5
    weights = tf.compat.v1.get_variable('weights', kernel_shape, initializer=weight_init_fn(stddev=std))
    biases = tf.compat.v1.get_variable('biases', [kernel_shape[-1]], initializer=bias_init_fn(bias_init_val))
    conv = tf.nn.conv2d(input=inp, filters=weights, strides=stride, padding=pad) + biases
    if dorelu:
      return tf.nn.relu(conv)
    else:
      return conv

def pool(inp, name=None, kernel=[2,2], stride=[2,2]):
  # Initialize max-pooling layer (default 2x2 window, stride 2)
  kernel = [1] + kernel + [1]
  stride = [1] + stride + [1]
  return tf.nn.max_pool2d(input=inp, ksize=kernel, strides=stride, padding='SAME', name=name)

def hourglass(inp, n, f, hg_id, attnum):
  nf = f
  up1 = cnv(inp, [3, 3, f, f], '%d_%d_up1' % (hg_id, n))
  up1_2 = up1
  pool1 = pool(inp, '%d_%d_pool' % (hg_id, n))
  low1 = cnv(pool1, [3, 3, f, nf], '%d_%d_low1' % (hg_id, n))
  if n > 1:
    low2 = hourglass(low1, n - 1, nf, hg_id,attnum)
  else:
    low2 = cnv(low1, [3, 3, nf, nf], '%d_%d_low2' % (hg_id, n))
  low3 = cnv(low2, [3, 3, nf, f], '%d_%d_low3' % (hg_id, n))
  up_size = tf.shape(input=up1)[1:3]
  up2 = tf.image.resize(low3, up_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return att(up1_2,tf.concat([up2,up1_2],axis=1), "%d_%d_att" % (hg_id, n,)) + up2

def build_from_gene(x, n_out_channel: int, seq: List[bool]):
  n_stack = geneop.cvtlstint(seq)
  f = 256
  x = tf.transpose(x, (0, 2, 1, 3))
  cnv1 = cnv(x, [7, 7, 1, 64], 'cnv1', stride=[1,1,1,1])
  cnv2 = cnv(cnv1, [3, 3, 64, 128], 'cnv2')
  cnv2b = cnv(cnv2, [3, 3, 128, 128], 'cnv2b')
  cnv3 = cnv(cnv2b, [3, 3, 128, 128], 'cnv3')
  cnv4 = cnv(cnv3, [3, 3, 128, f], 'cnv4')
  inter=cnv4
  preds = []
  for i in range(n_stack):
    hg = hourglass(inter, 4, f, i,1)
    cnv5 = cnv(hg, [3, 3, f, f], 'cnv5_%d' % i)
    cnv6 = cnv(cnv5, [1, 1, f, f], 'cnv6_%d' % i)
    preds += [cnv(cnv6, [1, 1, f, n_out_channel], 'out_%d' % i, dorelu=False)]
    if i < 3:
      inter = inter + cnv(cnv6, [1, 1, f, f], 'tmp_%d' % i, dorelu=False) + cnv(preds[-1], [1, 1, n_out_channel, f], 'tmp_out_%d'%i, dorelu = False)
  return [tf.transpose(v, (0, 2, 1, 3)) for v in preds]
