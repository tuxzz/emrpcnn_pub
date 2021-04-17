import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os, sys
sys.path.append("../lib")
from mir_util import infer, to_spec, to_wav_file
import config as cfg
from common import *
import sys
sys.is_train = False

import simpleopt

n_feature = 5644 // 2

dataset_type = simpleopt.get("dataset")
mix_path_list = simpleopt.get_multi("input")
ckpt_step = int(simpleopt.get("step"))
downmix = simpleopt.get_switch("downmix")

ver = simpleopt.get("ver")
gene = simpleopt.get("gene")

with cfg.ConfigBoundary():
  net_config, ch_list = {
    "mus2": (cfg.MUS2FConfig, ("inst", "vocal")),
  }[dataset_type]
  n_ch = len(ch_list)

  # Model
  print("* Initialize network model")
  p_input = tf.compat.v1.placeholder(tf.float32, shape=(1, 64, n_feature, 1), name="p_input")
  v_pred = infer(p_input, 2, False)
  if isinstance(v_pred, list):
    v_pred = v_pred[-1]

  x_input = np.zeros((1, 64, n_feature, 1),dtype=np.float32)
  with tf.compat.v1.Session(config=cfg.sess_cfg) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    print("* Load checkpoint")
    ckpt_path = os.path.join(net_config.ckpt_path, "checkpoint-%d" % (ckpt_step,))
    tf.compat.v1.train.Saver().restore(sess, ckpt_path)
    print(" :Loaded: `%s`" % (ckpt_path,))

    for mix_path in mix_path_list:
      print("* Compute `%s`" % (mix_path,))
      mixed_wav_mc, sr_orig = loadWav(mix_path)
      mixed_wav_mc = np.atleast_2d(mixed_wav_mc.T)
      if downmix:
        mixed_wav_mc = np.sum(mixed_wav_mc, axis=0, keepdims=True)
      if sr_orig != 44100:
        mixed_wav_mc = sp.resample_poly(mixed_wav_mc, 44100, sr_orig, axis=1).astype(np.float32)
      n_w_ch, n_w = mixed_wav_mc.shape

      est_ch_list = []
      for i_w_ch in range(n_w_ch):
        print("* Proc: Wave channel %d" % (i_w_ch,))
        mixed_wav = mixed_wav_mc[i_w_ch, :]

        mixed_spec = to_spec(mixed_wav, len_frame=5644, len_hop=1411)
        mixed_spec_mag = np.abs(mixed_spec)
        mixed_spec_phase = np.angle(mixed_spec)
        max_temp = np.max(mixed_spec_mag)
        mixed_spec_mag /= max_temp

        src_len = mixed_spec_mag.shape[0]
        start_idx = 0
        y_est = np.zeros((n_ch, src_len, n_feature), dtype=np.float32)
        while start_idx + 64 < src_len:
          x_input[0, :, :, 0] = mixed_spec_mag[start_idx:start_idx + 64, :n_feature]
          y_output = sess.run(v_pred, feed_dict={p_input: x_input})
          if start_idx == 0:
            for i_ch in range(n_ch):
              y_est[i_ch, start_idx:start_idx + 64, :] = y_output[0, :, :, i_ch]
          else:
            for i_ch in range(n_ch):
              y_est[i_ch, start_idx + 16:start_idx + 48, :] = y_output[0, 16:48, :, i_ch]
          start_idx += 32

        x_input[0, :, :, 0] = mixed_spec_mag[src_len - 64:src_len, 0:n_feature]
        y_output = sess.run(v_pred, feed_dict={p_input: x_input})
        src_start = src_len - start_idx - 16
        for i_ch in range(n_ch):
          y_est[i_ch, start_idx + 16:src_len, :] = y_output[0, 64 - src_start:64, :, i_ch]

        y_est *= max_temp
        l = []
        for i_ch, ch_name in enumerate(ch_list):
          y_wav = to_wav_file(y_est[i_ch, :, :], mixed_spec_phase[:, :n_feature], len_hop=1411)
          l.append(y_wav[np.newaxis, np.newaxis, :])
        l = np.concatenate(l, axis=0)
        est_ch_list.append(l)
      print("* Save")
      out = np.concatenate(est_ch_list, axis=1)
      for i_ch, ch_name in enumerate(ch_list):
        os.makedirs("split_out_f/{}_{}_step{}/".format(ver, gene, ckpt_step), exist_ok=True)
        wav = np.clip(np.round(out[i_ch, :, :].T * 32767), -32768, 32767).astype(np.int16)
        saveWav("split_out_f/{}_{}_step{}/{}_{}.wav".format(ver, gene, ckpt_step, ".".join(os.path.splitext(os.path.split(mix_path)[-1])[:-1]), ch_name), wav, 44100)
