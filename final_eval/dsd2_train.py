import tensorflow as tf
import os, time
import numpy as np
import random
from mir_util import infer
import pickle
import config as cfg

with cfg.ConfigBoundary():
  # Load data
  batch_size = cfg.DSD2Config.batch_size
  cache_path = cfg.dsd2_cache_path
  n_feature = cfg.frame_size // 2

  assert os.path.exists(cache_path), "Dataset cache not found"
  print("* Read cached spectrograms")
  mixed_list, vocal_list, inst_list, n_sample = pickle.load(open(cache_path, "rb"))
  print("* Number of training examples: %d" % (n_sample,))

  # Model
  print("* Initialize network")
  tf.compat.v1.random.set_random_seed(0x41526941)
  p_input = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, n_feature, 1), name="p_input")
  p_target = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, n_feature, 2), name="p_target")
  v_pred = infer(p_input, 2, True)
  if isinstance(v_pred, list):
    v_loss = 0
    for y in v_pred:
      v_loss += tf.reduce_mean(input_tensor=tf.abs(p_target - (y * p_input)))
  else:
    v_pred *= p_input
    v_loss = tf.reduce_mean(input_tensor=tf.abs(p_target - v_pred))
  # Loss, Optimizer
  v_global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="v_global_step")
  p_lr_fac = tf.compat.v1.placeholder(tf.float32, name="p_lr_fac")
  v_lr = p_lr_fac * tf.compat.v1.train.cosine_decay_restarts(cfg.DSD2Config.max_lr, v_global_step, cfg.DSD2Config.first_decay_period, alpha=cfg.DSD2Config.min_lr / cfg.MIR2Config.max_lr, t_mul=2.0)
  op_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=v_lr).minimize(v_loss, global_step=v_global_step)

  display_interval = 500
  loss_list = []
  rand_perm = np.random.permutation(n_sample)
  curr_idx = 0
  x_input = np.zeros((batch_size, 64, n_feature, 1), dtype=np.float32)
  y_input = np.zeros((batch_size, 64, n_feature, 2), dtype=np.float32)
  with tf.compat.v1.Session(config=cfg.sess_cfg) as sess:
    # Initialized, Load state
    sess.run(tf.compat.v1.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(cfg.DSD2Config.ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
      print("* Load checkpoint")
      tf.compat.v1.train.Saver().restore(sess, ckpt.model_checkpoint_path)

    t = time.time()
    for step in range(v_global_step.eval(), cfg.DSD2Config.final_step):
      random.seed(step)
      np.random.seed(step)
      tf.compat.v1.random.set_random_seed(step)
      for i in range(batch_size):
        seq = rand_perm[curr_idx]
        start = random.randint(0, mixed_list[seq].shape[0] - 64)
        x_input[i, :, :, 0] = mixed_list[seq][start:start + 64, :n_feature]
        y_input[i, :, :, 0] = inst_list[seq][start:start + 64, :n_feature]
        y_input[i, :, :, 1] = vocal_list[seq][start:start + 64, :n_feature]
        curr_idx += 1
        if curr_idx == n_sample:
          curr_idx = 0
          rand_perm = np.random.permutation(n_sample)

      if step <= 1000:
        lr_fac = 0.3 # Slow start to prevent some fast values go broken
      else:
        lr_fac = 1.0
      loss, _ = sess.run([v_loss, op_optim], feed_dict={p_input: x_input, p_target: y_input, p_lr_fac: lr_fac})

      loss_list.append(loss)
      if step % display_interval == 0:
        now = time.time()
        mean_loss = np.mean(loss_list)
        print("[%06d] loss=%.9f, d_time=%.3f" % (step, mean_loss, now - t))
        t = time.time()
        loss_list = []

      # Save state
      if step % cfg.DSD2Config.ckpt_step == 0:
        tf.compat.v1.train.Saver().save(sess, os.path.join(cfg.DSD2Config.ckpt_path, "checkpoint"), global_step=step)
