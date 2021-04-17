import os, sys
sys.path.append("../lib")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "1095919937"

import random
from common import *
import config as cfg
import librosa
import pickle, sys, os
import geneop, netop

n_gen = 100
n_out_channel = 2

train_seg_list = []
eval_seg_list = []
def eval_gene_core(gene):
  import time
  import tensorflow as tf
  import numpy as np
  from eval_util import bss_eval_sdr
  print(" :GENE: %d" % (geneop.cvtlstint(gene),))
  n_feature = cfg.n_feature
  tf.compat.v1.reset_default_graph()
  graph = tf.Graph()
  t = time.time()

  with graph.as_default():
    random.seed(0x41526941)
    np.random.seed(0x41526941)
    tf.compat.v1.random.set_random_seed(0x41526941)
    sess_conf = tf.compat.v1.ConfigProto(
      gpu_options=tf.compat.v1.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=1.0
      ),
      allow_soft_placement = True,
    )
    with tf.compat.v1.Session(config=sess_conf) as sess:
      # TRAIN
      p_feature = tf.compat.v1.placeholder(tf.float32, shape=(cfg.batch_size, cfg.n_hop_per_sample, n_feature, 1), name='x_mixed')
      p_target = tf.compat.v1.placeholder(tf.float32, shape=(cfg.batch_size, cfg.n_hop_per_sample, n_feature, n_out_channel), name='y_mixed')
      v_pred = geneop.build_from_gene(p_feature, n_out_channel, gene)

      n_param = netop.count_parameter()
      print(" :Total {:,} parameters in total".format(n_param))
      if "neg_gflops" in cfg.result_format:
        n_forward_flop = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()).total_float_ops
        print(" :Forward operation needs {:,} FLOPS".format(n_forward_flop))

      v_pred_clipped = tf.clip_by_value(v_pred, 0.0, 1.0) * p_feature
      v_loss = tf.reduce_mean(input_tensor=tf.abs(v_pred * p_feature - p_target))

      v_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="step")
      p_lr_fac = tf.compat.v1.placeholder(tf.float32, name="p_lr_fac")
      v_lr = p_lr_fac * tf.compat.v1.train.cosine_decay_restarts(cfg.max_lr, v_step, cfg.first_lr_period, alpha=cfg.min_lr / cfg.max_lr, t_mul=2.0)
      op_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=v_lr).minimize(v_loss, global_step=v_step)
      
      sess.run(tf.compat.v1.global_variables_initializer())
      loss_list = []
      data_feature = np.zeros((cfg.batch_size, cfg.n_hop_per_sample, n_feature, 1))
      data_target = np.zeros((cfg.batch_size, cfg.n_hop_per_sample, n_feature, n_out_channel))
      seg_idx_list = np.array([])
      for i_step in range(cfg.n_step):
        for i_batch in range(cfg.batch_size):
          if seg_idx_list.size == 0:
            seg_idx_list = np.random.permutation(len(train_seg_list))
          idx = seg_idx_list[0]
          seg_idx_list = seg_idx_list[1:]
          spec_mixed, spec_vocal, spec_inst = train_seg_list[idx]
          start_idx = np.random.randint(0, len(spec_mixed) - cfg.n_hop_per_sample)
          data_feature[i_batch, :, :, 0] = spec_mixed[start_idx:start_idx + cfg.n_hop_per_sample, :]
          data_target[i_batch, :, :, 0] = spec_inst[start_idx:start_idx + cfg.n_hop_per_sample, :]
          data_target[i_batch, :, :, 1] = spec_vocal[start_idx:start_idx + cfg.n_hop_per_sample, :]
        if i_step <= cfg.warmup_period:
          lr_fac = cfg.warmup_fac # Slow start to prevent some fast values go broken
        else:
          lr_fac = 1.0
        loss_value, _ = sess.run([v_loss, op_optim], feed_dict={p_feature: data_feature, p_target: data_target, p_lr_fac: lr_fac})
        loss_list.append(loss_value)
      # EVAL
      sdr_list = []
      valid_sdr_list = []
      ret_list = []
      for i_eval, (real_vocal, real_inst, magn_orig_list, phase_list, norm_fac) in enumerate(eval_seg_list):
        n_hop, _ = magn_orig_list.shape
        magn_inst_list = np.zeros_like(magn_orig_list, dtype=np.float32)
        magn_vocal_list = np.zeros_like(magn_orig_list, dtype=np.float32)
        data_feature = np.zeros((cfg.batch_size, cfg.n_hop_per_sample, n_feature, 1), dtype=np.float32)
        batch_hop_list = []
        def flush_buffer():
          pred_value, = sess.run([v_pred_clipped], feed_dict={p_feature: data_feature})
          for i_batch, (i_batch_hop, offset_begin, offset_end) in enumerate(batch_hop_list):
            magn_inst_list[i_batch_hop + offset_begin:i_batch_hop + offset_end, :-1] = pred_value[i_batch, offset_begin:offset_end, :, 0]
            magn_vocal_list[i_batch_hop + offset_begin:i_batch_hop + offset_end, :-1] = pred_value[i_batch, offset_begin:offset_end, :, 1]
          data_feature.fill(0.0)
          batch_hop_list.clear()
        def enqueue_buffer(data, i_batch_hop, offset_begin, offset_end):
          if len(batch_hop_list) == cfg.batch_size:
            flush_buffer()
          i_batch = len(batch_hop_list)
          data_feature[i_batch, :data.shape[0], :, 0] = data
          batch_hop_list.append((i_batch_hop, offset_begin, offset_end))
        i_hop = 0
        while i_hop + cfg.n_hop_per_sample < n_hop:
          data = magn_orig_list[i_hop:i_hop + cfg.n_hop_per_sample, :-1]
          if i_hop == 0:
            enqueue_buffer(data, i_hop, 0, cfg.n_hop_per_sample * 3 // 4)
          else:
            enqueue_buffer(data, i_hop, cfg.n_hop_per_sample // 4, cfg.n_hop_per_sample * 3 // 4)
          i_hop += cfg.n_hop_per_sample // 2
        data = magn_orig_list[i_hop:, :-1]
        enqueue_buffer(data, i_hop, cfg.n_hop_per_sample // 4, n_hop - i_hop)
        flush_buffer()
        unit_magn = np.exp(1j * phase_list)
        fake_inst = librosa.istft((magn_inst_list * unit_magn * norm_fac).T, hop_length=cfg.hop_size)
        fake_vocal = librosa.istft((magn_vocal_list * unit_magn * norm_fac).T, hop_length=cfg.hop_size)
        if (fake_inst <= 1e-8).all() or (fake_vocal <= 1e-8).all():
          sdr_list.append(-999999)
        else:
          #saveWav("fakeinst.wav", fake_inst, cfg.work_sr)
          #saveWav("fakevocal.wav", fake_vocal, cfg.work_sr)
          ret_list.append(cfg.pool.apply_async(bss_eval_sdr, (np.array([real_inst], dtype=np.float32), np.array([fake_inst], dtype=np.float32),)))
          ret_list.append(cfg.pool.apply_async(bss_eval_sdr, (np.array([real_vocal], dtype=np.float32), np.array([fake_vocal], dtype=np.float32),)))
      ret_list = [x.get()[0] for x in ret_list]
      for i_eval, sdr in enumerate(zip(ret_list[::2], ret_list[1::2])):
        mean_sdr = np.mean(sdr)
        if i_eval < cfg.n_eval:
          sdr_list.append(mean_sdr)
        else:
          valid_sdr_list.append(mean_sdr)

      result_list = []
      for result_type in cfg.result_format:
        if result_type == "sdr":
          result_list.append(np.mean(sdr_list))
        elif result_type == "neg_mega_pc":
          result_list.append(-n_param / 1000_000.0)
        elif result_type == "neg_gflops":
          result_list.append(-n_forward_flop / 1_000_000_000.0)
        elif result_type == "valid_sdr":
          result_list.append(np.mean(valid_sdr_list))
        else:
          raise ValueError("Unsupported result_type `%s`" % (result_type,))
      print("  EVAL RESULT: t=%.2f, train_loss=%.09f, result=%r" % (time.time() - t, np.mean(loss_list), result_list))
      return result_list

def eval_gene(gene):
  # noise reduction
  i_sdr = cfg.result_format.index("sdr")
  result_buf = [eval_gene_core(gene)]
  out = None
  while True:
    r = eval_gene_core(gene)
    for x in result_buf:
      if abs(r[i_sdr] - x[i_sdr]) < 0.2:
        real_sdr = (r[i_sdr] + x[i_sdr]) / 2.0
        out = [real_sdr if name == "sdr" else value for name, value in zip(cfg.result_format, r)]
        break
    if out is None:
      result_buf.append(r)
    else:
      break
  return out

def init_data():
  train_seg_list.clear()
  eval_seg_list.clear()
  import os, random
  import numpy as np
  import scipy.signal as sp
  print("* Prepare data")
  random.seed(0x41526941)
  np.random.seed(0x41526941)
  for (_, dir_list, _) in os.walk(os.path.join(cfg.dsd_root_path, "Mixtures", "Dev"), followlinks=True):
    for i_file, dirname in enumerate(sorted(dir_list)):
      if len(train_seg_list) == 0:
        print("* Train set")
      elif len(train_seg_list) == cfg.n_train and len(eval_seg_list) == 0:
        print("* Test set")
      elif len(eval_seg_list) == cfg.n_eval:
        print("* Validation set")
      print("  %04d:%s" % (i_file, dirname,))
      path_vocal = os.path.join(cfg.dsd_root_path, "Sources", "Dev", dirname, "vocals.wav")
      path_mix = os.path.join(cfg.dsd_root_path, "Mixtures", "Dev", dirname, "mixture.wav")
      w_mixed, sr_0 = loadWav(path_mix)
      w_vocal, sr = loadWav(path_vocal)
      assert sr_0 == sr
      w_mixed = np.mean(w_mixed, axis=1)
      w_vocal = np.mean(w_vocal, axis=1)
      w_inst = w_mixed - w_vocal

      w_mixed = sp.resample_poly(w_mixed, cfg.work_sr, sr).astype(np.float32)
      w_inst = sp.resample_poly(w_inst, cfg.work_sr, sr).astype(np.float32)
      w_vocal = sp.resample_poly(w_vocal, cfg.work_sr, sr).astype(np.float32)
      n_w, = w_mixed.shape
      sr = cfg.work_sr
      spec_mixed = librosa.stft(w_mixed, n_fft=cfg.fft_size, hop_length=cfg.hop_size).T
      if len(train_seg_list) != cfg.n_train:
        spec_mixed = np.abs(spec_mixed[:, :-1], dtype=np.float32)
        norm_fac = np.max(spec_mixed)
        spec_mixed /= norm_fac
        spec_inst = np.abs(librosa.stft(w_inst, n_fft=cfg.fft_size, hop_length=cfg.hop_size).T[:, :-1], dtype=np.float32) / norm_fac
        spec_vocal = np.abs(librosa.stft(w_vocal, n_fft=cfg.fft_size, hop_length=cfg.hop_size).T[:, :-1], dtype=np.float32) / norm_fac
        train_seg_list.append((spec_mixed, spec_vocal, spec_inst))
      elif len(eval_seg_list) < cfg.n_eval + cfg.n_valid:
        spec_magn = np.abs(spec_mixed, dtype=np.float32)
        norm_fac = np.max(spec_magn)
        spec_magn /= norm_fac
        eval_seg_list.append((w_vocal, w_inst, spec_magn, np.angle(spec_mixed), norm_fac))
        if len(eval_seg_list) == cfg.n_eval + cfg.n_valid:
            break
      else:
        assert False
  assert len(train_seg_list) == cfg.n_train, "No enough data for training"
  assert len(eval_seg_list) == cfg.n_eval + cfg.n_valid, "No enough data for evaluating"

def pre_fn():
  cfg.result_format = cfg.worker_config["result_format"]
  cfg.batch_size = cfg.worker_config["batch_size"]
  cfg.max_lr = cfg.worker_config["max_lr"]
  cfg.min_lr = cfg.worker_config["min_lr"]
  cfg.warmup_fac = cfg.worker_config["warmup_fac"]
  cfg.first_lr_period = cfg.worker_config["first_lr_period"]
  cfg.warmup_period = cfg.worker_config["warmup_period"]
  cfg.n_hop_per_sample = cfg.worker_config["n_hop_per_sample"]
  cfg.work_sr = cfg.worker_config["work_sr"]
  cfg.hop_size = cfg.worker_config["hop_size"]
  cfg.fft_size = cfg.worker_config["fft_size"]
  cfg.n_feature = cfg.worker_config["n_feature"]
  cfg.n_step = cfg.worker_config["n_step"]
  cfg.n_train = cfg.worker_config["n_train"]
  cfg.n_eval = cfg.worker_config["n_eval"]
  cfg.n_valid = cfg.worker_config.get("n_valid", 0)
  print("[CONFIG] hop_size=%d, n_feature=%d" % (cfg.hop_size, cfg.n_feature))
  init_data()

def clean_env():
  train_seg_list.clear()
  eval_seg_list.clear()
  try:
    del cfg.result_format
    del cfg.batch_size
    del cfg.max_lr, cfg.min_lr, cfg.warmup_fac, cfg.first_lr_period, cfg.warmup_period
    del cfg.n_hop_per_sample
    del cfg.work_sr
    del cfg.hop_size
    del cfg.fft_size
    del cfg.n_feature
    del cfg.n_step
    del cfg.n_train
    del cfg.n_eval
    del cfg.n_valid
  except:
    pass

def eval_fn(gene):
  return eval_gene(gene)