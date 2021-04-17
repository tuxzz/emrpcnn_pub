def worker_main():
  import os
  if os.name == "posix":
    os.nice(19)

def main():
  import multiprocessing as mp
  import numpy as np
  import tensorflow as tf
  import os, sys
  from mir_util import infer, to_spec, to_wav_file
  import scipy.signal as sp
  import config as cfg
  sys.path.append("../lib")
  from eval_util import bss_eval
  from common import loadWav
  import redirect, simpleopt

  step_idx = int(simpleopt.get("step"))
  n_eval = simpleopt.get("first", None)
  with cfg.ConfigBoundary():
    batch_size = 1
    n_feature = cfg.frame_size // 2

    # Model
    print("* Initialize network")
    p_input = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, n_feature, 1), name="p_input")
    v_pred = infer(p_input, 2, False)
    if isinstance(v_pred, list):
      v_pred = v_pred[-1]
    v_pred = tf.clip_by_value(v_pred, 0.0, 1.0) * p_input

    x_input = np.zeros((batch_size, 64, n_feature, 1), dtype=np.float32)
    with tf.compat.v1.Session(config=cfg.sess_cfg) as sess:
      # Initialized, Load state
      sess.run(tf.compat.v1.global_variables_initializer())
      
      print("* Load checkpoint")
      ckpt_path = os.path.join(cfg.MIR2Config.ckpt_path, "checkpoint-%d" % (step_idx,))
      tf.compat.v1.train.Saver().restore(sess, ckpt_path)
      print(" :Loaded: `%s`" % (ckpt_path,))

      os.makedirs("./eval_output", exist_ok=True)
      name_list = []
      ret_list = []
      with mp.Pool(processes=1, initializer=worker_main) as pool:
        for (root, _, file_list) in os.walk(cfg.mir_wav_path):
          file_list = sorted(f for f in file_list if not (f.startswith("abjones") or f.startswith("amy")))
          if n_eval is not None:
            file_list = file_list[:int(n_eval)]
          for i_file, filename in enumerate(file_list):
            print("[%03d/%03d] SEND: `%s`" % (i_file + 1, len(file_list), filename,))
            name_list.append(filename)
            path = os.path.join(root, filename)

            mixed_wav, sr_orig = loadWav(path)
            gt_wav_vocal = mixed_wav[:, 1]
            gt_wav_inst = mixed_wav[:, 0]
            mixed_wav = np.sum(mixed_wav, axis=1)

            mixed_wav_orig = mixed_wav
            gt_wav_vocal_orig = gt_wav_vocal
            gt_wav_inst_orig = gt_wav_inst

            gt_wav_vocal = sp.resample_poly(gt_wav_vocal, cfg.sr, sr_orig).astype(np.float32)
            gt_wav_inst = sp.resample_poly(gt_wav_inst, cfg.sr, sr_orig).astype(np.float32)
            mixed_wav = sp.resample_poly(mixed_wav, cfg.sr, sr_orig).astype(np.float32)

            mixed_spec = to_spec(mixed_wav)
            mixed_spec_mag = np.abs(mixed_spec)
            mixed_spec_phase = np.angle(mixed_spec)
            max_tmp = np.max(mixed_spec_mag)
            mixed_spec_mag = mixed_spec_mag / max_tmp

            src_len = mixed_spec_mag.shape[0]
            start_idx = 0
            y_est_inst = np.zeros((src_len, n_feature),dtype=np.float32)
            y_est_vocal = np.zeros((src_len, n_feature),dtype=np.float32)
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
            y_wav_inst = to_wav_file(y_est_inst, mixed_spec_phase[:, :n_feature])
            y_wav_vocal = to_wav_file(y_est_vocal, mixed_spec_phase[:, :n_feature])
            #saveWav("inst.wav", y_wav_inst, cfg.sr)
            #saveWav("vocal.wav", y_wav_vocal, cfg.sr)

            # upsample to original samprate
            y_wav_inst_orig = sp.resample_poly(y_wav_inst, sr_orig, cfg.sr).astype(np.float32)
            y_wav_vocal_orig = sp.resample_poly(y_wav_vocal, sr_orig, cfg.sr).astype(np.float32)
            ret_list.append(pool.apply_async(bss_eval, (mixed_wav_orig, gt_wav_inst_orig, gt_wav_vocal_orig, y_wav_inst_orig, y_wav_vocal_orig,)))
        with redirect.ConsoleAndFile("./eval_output/mir2_%s_%d_step%d.txt" % (cfg.gene_ver, cfg.gene_value, step_idx)) as r:
          gnsdr = 0.0
          gsir = 0.0
          gsar = 0.0
          total_len = 0
          for name, ret in zip(name_list, ret_list):
            nsdr, sir, sar, lens = ret.get()
            printstr = name + " " + str(nsdr) + " " + str(sir) + " " + str(sar)
            r.print(printstr)
            total_len += lens
            gnsdr += nsdr * lens
            gsir += sir * lens
            gsar += sar * lens
          r.print("Final results")
          r.print("GNSDR [Accompaniments, voice]")
          r.print(gnsdr / total_len)
          r.print("GSIR [Accompaniments, voice]")
          r.print(gsir / total_len)
          r.print("GSAR [Accompaniments, voice]")
          r.print(gsar / total_len)

if __name__ == "__main__":
  main()
