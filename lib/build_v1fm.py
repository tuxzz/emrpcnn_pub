import sys
import tensorflow as tf

from typing import List, Dict, Tuple, Callable
import geneop, netop

gene_len = 142

def cb_from_gene(x, n_channel: int, seq: List[bool]):
  relu = netop.relu
  sigmoid = netop.sigmoid
  assert len(seq) == 1
  # Segmentation
  seg_a = seq[0:1]
  # Translation
  act_fn = sigmoid if seg_a[0] else relu
  # Apply
  return netop.ret_new(act_fn(netop.conv2d(x, (3, 3), n_channel)))

def ccg_from_gene(x, n_channel: int, seq: List[bool]):
  assert len(seq) == 3
  # Segmentation
  seg_a = seq[0:1]
  seg_cb_0 = seq[1:2]
  seg_cb_1 = seq[2:3]
  # Translation
  use_res = seg_a[0]
  # Apply
  v = x
  v = cb_from_gene(v, n_channel, seg_cb_0)
  vv = cb_from_gene(v, n_channel, seg_cb_1)

  return netop.ret_new(netop.add(v, vv) if use_res else vv)

def cg_from_gene(x, seq: List[bool]):
  assert len(seq) == 5
  cnv_channel_table = (None, 32, 64, 128)
  # Segmentation
  seg_a = seq[0:2]
  ccg_0 = seq[2:5]
  # Translation
  n_channel = cnv_channel_table[geneop.cvtlstgray(seg_a)]
  # Apply
  if n_channel is None:
    return x
  else:
    return ccg_from_gene(x, n_channel, ccg_0)

def pb_from_gene(x, seq: List[bool]):
  assert len(seq) == 4
  size_table = (1, 4, 16, 64)
  # Segmentation
  seg_h = seq[0:2]
  seg_w = seq[2:4]
  # Translation
  h_size = size_table[geneop.cvtlstgray(seg_h)]
  w_size = size_table[geneop.cvtlstgray(seg_w)]
  # Apply
  v = x
  if h_size == 1 and w_size == 1:
    return v
  v = netop.adv_pool(v, (h_size, w_size), (h_size, w_size), ("boxcar", "boxcar"), False)
  return netop.ret_new(v)

def rb_from_gene(x, seq: List[bool]):
  assert len(seq) == 9
  cnv_size_table = (16, 32, 64, 128)
  # Segmentation
  pb_0 = seq[0:4]
  seg_a = seq[4:6]
  ccg_0 = seq[6:9]
  # Translation
  cnv_size = cnv_size_table[geneop.cvtlstgray(seg_a)]
  # Apply
  v = x
  vv = pb_from_gene(v, pb_0)
  if vv is v:
    return v
  v = vv
  v = ccg_from_gene(v, cnv_size, ccg_0)
  v = netop.resize_nearest(v, netop.get_shape(x)[1:3])
  return netop.ret_new(v)

def b_from_gene(x, n_out_channel: int, seq: List[bool]):
  assert len(seq) == 26
  # Segmentation
  cg_0 = seq[0:5]
  rb_0 = seq[5:14]
  rb_1 = seq[14:23]
  ccg_0 = seq[23:26]
  # Apply
  v = x
  v = cg_from_gene(v, cg_0)
  v_rb_0 = rb_from_gene(v, rb_0)
  v_rb_1 = rb_from_gene(v, rb_1)
  l = [v]
  if v_rb_0 is not v:
    l.append(v_rb_0)
  if v_rb_1 is not v:
    l.append(v_rb_1)
  if len(l) > 1:
    v = netop.concat(l, axis=-1)
  elif len(l) == 1:
    v = l[0]
  v = ccg_from_gene(v, n_out_channel, ccg_0)
  return netop.ret_new(v)

def build_from_gene(x, n_out_channel: int, seq: List[bool]):
  import numpy as np
  assert len(seq) == 142
  cnv_channel_table = (32, 64, 128, 256)
  # Segmentation
  seg_a = seq[0:2]
  seg_b = seq[2:12]
  b_0 = seq[12:38]
  b_1 = seq[38:64]
  b_2 = seq[64:90]
  b_3 = seq[90:116]
  b_4 = seq[116:142]
  # Translation
  n_channel = cnv_channel_table[geneop.cvtlstgray(seg_a)]
  MRL = seg_b # b-bb-bbb-bbbb | 0-12-345-6789
  # Apply
  v = x # (batch_size, n_frame, n_spec, n_channel)
  
  batch_size, n_frame, n_spec, n_in_ch = [int(x) for x in v.get_shape()]
  v = v[:, :, :512, :]
  
  b0 = b_from_gene(v, n_channel, b_0)

  b1 = b_from_gene(b0, n_channel, b_1)
  b1 = netop.add(b1, b0) if MRL[0] else b1

  b2 = b_from_gene(b1, n_channel, b_2)
  b2 = netop.add(b2, b0) if MRL[1] else b2
  b2 = netop.add(b2, b1) if MRL[2] else b2

  b3 = b_from_gene(b2, n_channel, b_3)
  b3 = netop.add(b3, b0) if MRL[3] else b3
  b3 = netop.add(b3, b1) if MRL[4] else b3
  b3 = netop.add(b3, b2) if MRL[5] else b3

  b4 = b_from_gene(b3, n_channel, b_4)
  b4 = netop.add(b4, b0) if MRL[6] else b4
  b4 = netop.add(b4, b1) if MRL[7] else b4
  b4 = netop.add(b4, b2) if MRL[8] else b4
  b4 = netop.add(b4, b3) if MRL[9] else b4
  
  v = netop.ret_new(netop.conv2d(b4, (1, 1), n_out_channel)) # (batch_size, n_frame, n_feature, n_channel)
  
  l = []
  sgv = tf.stop_gradient(v)
  for i_ch in range(n_out_channel):
    vv = tf.keras.layers.Dense(n_spec - 512, activation="relu", use_bias=True, kernel_initializer=tf.compat.v1.initializers.he_normal(seed=0x12345678)).apply(sgv[:, :, :, i_ch])
    vv = tf.reshape(vv, (batch_size, n_frame, n_spec - 512, 1,))
    l.append(vv)
  vv = tf.concat(l, axis=3)
  v = tf.concat([v, vv], axis=2)
  
  if sys.is_train:
    v = v * x
  else:
    v = tf.clip_by_value(v, 0.0, 1.0) * x
  
  return v
