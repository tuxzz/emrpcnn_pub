def proc_one(mixed_wav_multich, vocals_wav_multich, i_ch, i_aug, sr_orig, sr, hop_size, frame_size):
  import sys, os, pickle
  sys.path.append("../lib")
  from mir_util import to_spec, rndshift
  import scipy.signal as sp
  import numpy as np
  import random
  import gc
  
  if i_ch == 2:
    mixed_wav = np.sum(mixed_wav_multich, axis=1)
    vocals_wav = np.sum(vocals_wav_multich, axis=1)
  else:
    mixed_wav = mixed_wav_multich[:, i_ch]
    vocals_wav = vocals_wav_multich[:, i_ch]
  del mixed_wav_multich, vocals_wav_multich
  gc.collect()
  if sr != sr_orig:
    mixed_wav = sp.resample_poly(mixed_wav, sr, sr_orig).astype(np.float32)
    vocals_wav = sp.resample_poly(vocals_wav, sr, sr_orig).astype(np.float32)
  inst_wav = mixed_wav - vocals_wav
  if i_aug != 0:
    vocals_wav = rndshift(vocals_wav, sr // 2)
    inst_wav = rndshift(inst_wav, sr // 2)
    vocals_wav *= np.random.uniform(0.5, 1.5)
    mixed_wav = inst_wav + vocals_wav
  mixed_spec = to_spec(mixed_wav, len_frame=frame_size, len_hop=hop_size)
  mixed_spec_mag = np.abs(mixed_spec)
  vocals_spec = to_spec(vocals_wav, len_frame=frame_size, len_hop=hop_size)
  vocals_spec_mag = np.abs(vocals_spec)
  inst_spec = to_spec(inst_wav, len_frame=frame_size, len_hop=hop_size)
  inst_spec_mag = np.abs(inst_spec)
  max_tmp = np.max(mixed_spec_mag)
  mixed = mixed_spec_mag * 128.0 / max_tmp
  vocals = vocals_spec_mag * 128.0 / max_tmp
  inst = inst_spec_mag * 128.0 / max_tmp
  gc.collect()
  return mixed.astype(np.float16), vocals.astype(np.float16), inst.astype(np.float16)

def worker_main():
  import os
  if os.name == "posix":
    os.nice(19)

def mkcache_dirty(source_name):
  import sys, os, pickle
  import scipy.signal as sp
  sys.path.append("../lib")
  import simpleopt
  from common import loadWav
  import numpy as np
  import random
  import db
  import multiprocessing as mp
  import gc
  import config as cfg
  import ctypes
  
  if source_name == "vocals":
    cache_meta_path = "~/mus2f.cache.meta"
    cache_path = "~/mus2f.cache"
  else:
    cache_meta_path = "~/mus2f.{}.cache.meta".format(source_name)
    cache_path = "~/mus2f.{}.cache".format(source_name)
  
  with mp.Pool(processes=5, initializer=worker_main) as pool:
    n_sample = 0
    mixed_list = []
    inst_list = []
    vocal_list = []
    n_feature = 5644 // 2
    #n_aug = simpleopt.get("aug", default=4, ok=int)

    print("* Generate spectrograms")
    np.random.seed(0x41526941)
    random.seed(0x41526941)
    mus_dict = db.index_musdb(cfg.mus_train_path)
    n_total_frame = 0
    
    import ctypes
    madvise = ctypes.CDLL("libc.so.6").madvise

    curr_seg_pos = 0
    seg_list = []
    for i_song, song_name in enumerate(sorted(mus_dict.keys())):
      print("[Compute:{}]{}".format(i_song, song_name))
      filename_vocal = mus_dict[song_name][source_name]
      filename_mix = mus_dict[song_name]["mix"]
      mixed_wav_multich, sr_orig = loadWav(filename_mix)
      vocals_wav_multich, _ = loadWav(filename_vocal)
      ret_list = []
      for i_aug in range(2):
        for i_ch in range(3):
          ret_list.append(pool.apply_async(proc_one, (mixed_wav_multich, vocals_wav_multich, i_ch, i_aug, sr_orig, 44100, 5644 // 4, 5644,)))
          n_sample += 1
      for i, x in enumerate(ret_list):
        print("  Subsample {}".format(i))
        mixed, vocals, inst = x.get()
        n = mixed.shape[0]
        if os.path.isfile(os.path.expanduser(cache_path)):
          mode = "r+"
        else:
          mode = "w+"
        out = np.memmap(os.path.expanduser(cache_path), np.float16, mode, offset=0, shape=(curr_seg_pos + n, 2823, 3))
        madv_ret = madvise(out.ctypes.data, out.size * out.dtype.itemsize, 2) == 0 # 1 = MADV_RANDOM, 2 = MADV_SEQUENTIAL
        assert madv_ret == 0, "MADVISE FAILED"
        out[curr_seg_pos:curr_seg_pos + n, :, 0] = mixed
        out[curr_seg_pos:curr_seg_pos + n, :, 1] = vocals
        out[curr_seg_pos:curr_seg_pos + n, :, 2] = inst
        out.flush()
        del out
        curr_seg_pos += n
        seg_list.append(curr_seg_pos)
        del mixed, vocals, inst, x
        gc.collect()
    assert n_sample > 0
    
    print("* Write metadata")
    meta = {
      "seg_list": seg_list,
    }
    with open(os.path.expanduser(cache_meta_path), "wb") as f:
      pickle.dump(meta, f)
    
    print("* Cache generation is finished.")

def mkcache_main(source_name):
  import os, pickle
  import ctypes
  import numpy as np
  madvise = ctypes.CDLL("libc.so.6").madvise
  
  if source_name == "vocals":
    cache_meta_path = "~/mus2f.cache.meta"
    cache_path = "~/mus2f.cache"
  else:
    cache_meta_path = "~/mus2f.{}.cache.meta".format(source_name)
    cache_path = "~/mus2f.{}.cache".format(source_name)
    
  if not os.path.isfile(os.path.expanduser(cache_meta_path)):
    mkcache_dirty(source_name)
  else:
    print("* Memory mapping from cached data...")
  with open(os.path.expanduser(cache_meta_path), "rb") as f:
    meta = pickle.load(f)
  seg_list = meta["seg_list"]
  ss = np.memmap(os.path.expanduser(cache_path), np.float16, "r", offset=0, shape=(seg_list[-1], 2823, 3))
  madv_ret = madvise(ss.ctypes.data, ss.size * ss.dtype.itemsize, 1) == 0 # 1 = MADV_RANDOM, 2 = MADV_SEQUENTIAL
  assert madv_ret == 0, "MADVISE FAILED"
  
  mixed_list, vocal_list, inst_list, n_sample = [], [], [], len(seg_list)
  a = 0
  for b in seg_list:
    s = ss[a:b, :, :]
    mixed_list.append(s[:, :, 0])
    vocal_list.append(s[:, :, 1])
    inst_list.append(s[:, :, 2])
    a = b
  
  return mixed_list, vocal_list, inst_list, n_sample

def main():
  import sys
  sys.is_train = True
  sys.path.append("../lib")
  import simpleopt
  source_name = simpleopt.get("source", "vocals")
  mixed_list, vocal_list, inst_list, n_sample = mkcache_main(source_name)
  
  import tensorflow as tf
  import os, time
  import numpy as np
  import random
  from mir_util import infer
  import config as cfg

  with cfg.ConfigBoundary():
    ckpt_path = cfg.MUS2FConfig.ckpt_path
    if source_name != "vocals":
      ckpt_path = "{}_{}".format(ckpt_path, source_name)
    # Load data
    batch_size = cfg.MUS2FConfig.batch_size
    #cache_path = cfg.mus2f_cache_path
    n_feature = 5644 // 2

    #assert os.path.exists(cache_path), "Dataset cache not found"
    #print("* Read cached spectrograms")
    #mixed_list, vocal_list, inst_list, n_sample = joblib.load(cache_path, mmap_mode="r")
    print("* Number of training examples: %d" % (n_sample,))

    # Model
    print("* Initialize network")
    tf.compat.v1.random.set_random_seed(0x41526941)
    #tf.compat.v1.random.set_random_seed(0x41526942)
    p_input = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, n_feature, 1), name="p_input")
    p_target = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, n_feature, 2), name="p_target")
    v_pred = infer(p_input, 2, True)
    if isinstance(v_pred, list):
      v_loss = 0
      for y in v_pred:
        v_loss += tf.reduce_mean(input_tensor=tf.abs(p_target - y))
    else:
      v_loss = tf.reduce_mean(input_tensor=tf.abs(p_target - v_pred))
    # Loss, Optimizer
    v_global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="v_global_step")
    p_lr_fac = tf.compat.v1.placeholder(tf.float32, name="p_lr_fac")
    v_lr = p_lr_fac * tf.compat.v1.train.cosine_decay_restarts(cfg.MUS2FConfig.max_lr, v_global_step, cfg.MUS2FConfig.first_decay_period, alpha=cfg.MUS2FConfig.min_lr / cfg.MUS2FConfig.max_lr, t_mul=2.0)
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

      ckpt = tf.train.get_checkpoint_state(ckpt_path)
      if ckpt and ckpt.model_checkpoint_path:
        print("* Load checkpoint")
        tf.compat.v1.train.Saver().restore(sess, ckpt.model_checkpoint_path)

      t = time.time()
      for step in range(v_global_step.eval(), cfg.MUS2FConfig.final_step):
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
        x_input /= 128.0
        y_input /= 128.0
        
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
        if step % cfg.MUS2FConfig.ckpt_step == 0:
          tf.compat.v1.train.Saver().save(sess, os.path.join(ckpt_path, "checkpoint"), global_step=step)

if __name__ == "__main__":
  main()
