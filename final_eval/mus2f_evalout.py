def worker_main():
  import sys, os
  sys.path.append("../lib")
  if os.name == "posix":
    os.nice(19)

def main_estimate(pool):
  import numpy as np
  import tensorflow as tf
  import os, sys, pathlib
  sys.path.append("../lib")
  import config as cfg
  #import librosa
  from mir_util import infer, to_spec, to_wav_file
  import scipy.signal as sp
  import musdb, museval
  import simpleopt
  sys.is_train = False

  step_idx = int(simpleopt.get("step"))
  n_eval = simpleopt.get("first", None)
  if n_eval is not None:
    n_eval = int(n_eval)
    assert n_eval > 0
  
  sound_sample_root = simpleopt.get("sound-out", None)
  source = simpleopt.get("source")
  if source == "vocals":
    source = None
  
  with cfg.ConfigBoundary():
    if source is None:
      model_name = "mus2f_%s_%d_step%d" % (cfg.gene_ver, cfg.gene_value, step_idx,)
    else:
      model_name = "mus2f_%s_%d_step%d_%s" % (cfg.gene_ver, cfg.gene_value, step_idx, source,)
    model_name_nosrc = "mus2f_%s_%d_step%d" % (cfg.gene_ver, cfg.gene_value, step_idx,)
    if sound_sample_root is None:
      sound_sample_root = "./sound_output_mus2f/{}".format(model_name_nosrc)
    pathlib.Path(sound_sample_root).mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.MUS2FConfig.ckpt_path
    if source != "vocals":
      ckpt_path = "{}_{}".format(ckpt_path, source)
    
    batch_size = 1
    n_feature = 5644 // 2

    # Model
    print("* Initialize network")
    p_input = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, n_feature, 1), name="p_input")
    v_pred = infer(p_input, 2, False)
    if isinstance(v_pred, list):
      v_pred = v_pred[-1]

    with tf.compat.v1.Session(config=cfg.sess_cfg) as sess:
      # Initialized, Load state
      sess.run(tf.compat.v1.global_variables_initializer())
      
      print("* Load checkpoint")
      ckpt_path = os.path.join(ckpt_path, "checkpoint-%d" % (step_idx,))
      tf.compat.v1.train.Saver().restore(sess, ckpt_path)
      print(" :Loaded: `%s`" % (ckpt_path,))

      os.makedirs("./eval_output", exist_ok=True)
      name_list = []
      ret_list = []
      
      mus = musdb.DB(
        root=cfg.mus_root_path,
        download=False,
        subsets="test",
        is_wav=True
      )
      mus_trk_list = list(mus.tracks)
      mus_trk_list.sort(key=lambda x:x.name)
      assert len(mus_trk_list) > 0
      if n_eval is not None:
        mus_trk_list = mus_trk_list[:n_eval]
      
      results = museval.EvalStore()
      
      for i_song, track in enumerate(mus_trk_list):
        print("[%02d/%02d] Estimate: `%s`" % (i_song + 1, len(mus_trk_list), track.name,))
        voc_ch_list = []
        inst_ch_list = []
        for i_channel in range(2):
          print(" :Channel #%d" % (i_channel,))
          name_list.append(track.name + " Channel %d" % (i_channel,))

          mixed_wav = track.audio[:, i_channel]
          mixed_spec = to_spec(mixed_wav, len_frame=5644, len_hop=5644 // 4)
          mixed_spec_mag = np.abs(mixed_spec)
          mixed_spec_phase = np.angle(mixed_spec)
          max_tmp = np.max(mixed_spec_mag)
          mixed_spec_mag = mixed_spec_mag / max_tmp

          src_len = mixed_spec_mag.shape[0]
          start_idx = 0
          y_est_inst = np.zeros((src_len, n_feature), dtype=np.float32)
          y_est_vocal = np.zeros((src_len, n_feature), dtype=np.float32)
          x_input = np.zeros((batch_size, 64, n_feature, 1), dtype=np.float32)
          while start_idx + 64 < src_len:
            x_input[0, :, :, 0] = mixed_spec_mag[start_idx:start_idx + 64, :n_feature]
            y_output = sess.run(v_pred, feed_dict={p_input: x_input})
            if start_idx == 0:
              y_est_inst[start_idx:start_idx + 64,:] = y_output[0, :, :, 0]
              y_est_vocal[start_idx:start_idx + 64,:] = y_output[0, :, :, 1]
            else:
              y_est_inst[start_idx + 16:start_idx + 48, :] = y_output[0, 16:48, :, 0]
              y_est_vocal[start_idx + 16:start_idx + 48, :] = y_output[0, 16:48, :, 1]
            start_idx += 32

          x_input[0, :, :, 0] = mixed_spec_mag[src_len - 64:src_len, :n_feature]
          y_output = sess.run(v_pred, feed_dict={p_input: x_input})
          src_start = src_len - start_idx - 16
          y_est_inst[start_idx + 16:src_len, :] = y_output[0, 64 - src_start:64, :, 0]
          y_est_vocal[start_idx + 16:src_len, :] = y_output[0, 64 - src_start:64, :, 1]

          y_est_inst *= max_tmp
          y_est_vocal *= max_tmp
          y_wav_inst = to_wav_file(y_est_inst, mixed_spec_phase[:, :n_feature], len_hop=5644 // 4)
          y_wav_vocal = to_wav_file(y_est_vocal, mixed_spec_phase[:, :n_feature], len_hop=5644 // 4)
          
          voc_ch_list.append(y_wav_vocal.reshape(y_wav_vocal.size, 1))
          inst_ch_list.append(y_wav_inst.reshape(y_wav_inst.size, 1))
          del y_wav_inst, y_wav_vocal, y_est_inst, y_est_vocal, src_start, x_input, y_output, mixed_spec_mag, max_tmp, mixed_spec_phase, mixed_spec, mixed_wav
        estimates = {
          source: np.concatenate(voc_ch_list, axis=1),
        }
        del voc_ch_list, inst_ch_list
        if sound_sample_root:
          mus.save_estimates(estimates, track, sound_sample_root)
        del estimates, i_song, track

def main(pool):
  main_estimate(pool)

if __name__ == "__main__":
  import multiprocessing as mp
  with mp.Pool(processes=8, initializer=worker_main) as pool:
    main(pool)
