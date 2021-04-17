def worker_main():
  import os
  if os.name == "posix":
    os.nice(19)

def main():
  import multiprocessing as mp
  import numpy as np
  import tensorflow as tf
  import os, sys
  import config as cfg
  #import librosa
  from mir_util import infer, to_spec, to_wav_file
  import scipy.signal as sp
  sys.path.append("../lib")
  from eval_util import bss_eval_sdr
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
      ckpt_path = os.path.join(cfg.DSD2Config.ckpt_path, "checkpoint-%d" % (step_idx,))
      tf.compat.v1.train.Saver().restore(sess, ckpt_path)
      print(" :Loaded: `%s`" % (ckpt_path,))

      os.makedirs("./eval_output", exist_ok=True)
      name_list = []
      ret_list = []
      with mp.Pool(processes=1, initializer=worker_main) as pool:
        for (root, dir_list, _) in os.walk(os.path.join(cfg.dsd_path, "Mixtures", "Test")):
          dir_list = sorted(dir_list)
          if n_eval is not None:
            dir_list = dir_list[:int(n_eval)]
          for i_dir, d in enumerate(dir_list):
            print("[%02d/%02d] STG1: `%s`" % (i_dir + 1, len(dir_list), d,))
            name_list.append(d)

            filename_vocal = os.path.join(cfg.dsd_path, "Sources", "Test", d, "vocals.wav")
            filename_mix = os.path.join(cfg.dsd_path, "Mixtures", "Test", d, "mixture.wav")

            import time
            t = time.time()
            mixed_wav_orig, sr_orig = loadWav(filename_mix)#librosa.load(filename_mix, sr=None, mono=True)
            mixed_wav_orig = np.sum(mixed_wav_orig, axis=1)
            gt_wav_vocal_orig, _ = loadWav(filename_vocal)#librosa.load(filename_vocal, sr=None, mono=True)[0]
            gt_wav_vocal_orig = np.sum(gt_wav_vocal_orig, axis=1)
            gt_wav_inst_orig = mixed_wav_orig - gt_wav_vocal_orig

            mixed_wav = sp.resample_poly(mixed_wav_orig, cfg.sr, sr_orig).astype(np.float32)#librosa.load(filename_mix, sr=cfg.sr, mono=True)[0]
            gt_wav_vocal = sp.resample_poly(gt_wav_vocal_orig, cfg.sr, sr_orig).astype(np.float32)#librosa.load(filename_vocal, sr=cfg.sr, mono=True)[0]
            gt_wav_inst = mixed_wav - gt_wav_vocal
            mixed_spec = to_spec(mixed_wav)
            mixed_spec_mag = np.abs(mixed_spec)
            mixed_spec_phase = np.angle(mixed_spec)
            max_tmp = np.max(mixed_spec_mag)
            mixed_spec_mag = mixed_spec_mag / max_tmp

            src_len = mixed_spec_mag.shape[0]
            start_idx = 0
            y_est_inst = np.zeros((src_len, n_feature), dtype=np.float32)
            y_est_vocal = np.zeros((src_len, n_feature), dtype=np.float32)
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

            #upsample to original SR
            y_wav_inst_orig = sp.resample_poly(y_wav_inst, sr_orig, cfg.sr).astype(np.float32)#librosa.resample(y_wav_inst, cfg.sr, sr_orig)
            y_wav_vocal_orig = sp.resample_poly(y_wav_vocal, sr_orig, cfg.sr).astype(np.float32)#librosa.resample(y_wav_vocal, cfg.sr, sr_orig)
            
            ret_list.append(pool.apply_async(bss_eval_sdr, (np.array([gt_wav_inst_orig, gt_wav_vocal_orig], dtype=np.float32), np.array([y_wav_inst_orig, y_wav_vocal_orig], dtype=np.float32),)))
        with redirect.ConsoleAndFile("./eval_output/dsd2_%s_%d_step%d.txt" % (cfg.gene_ver, cfg.gene_value, step_idx)) as r:
          sdr_inst = []
          sdr_vocal = []
          sir_inst = []
          sir_vocal = []
          sar_inst = []
          sar_vocal = []
          for name, ret in zip(name_list, ret_list):
            r.print(name)
            sdr, sir, sar = ret.get()
            sdr_inst.append(sdr[0])
            sir_inst.append(sir[0])
            sar_inst.append(sar[0])
            sdr_vocal.append(sdr[1])
            sir_vocal.append(sir[1])
            sar_vocal.append(sar[1])
            printstr = " Inst SDR=%f, SIR=%f, SAR=%f, Vocal SDR=%f, SIR=%f, SAR=%f" % (sdr_inst[-1], sir_inst[-1], sar_inst[-1], sdr_vocal[-1], sir_vocal[-1], sar_vocal[-1])
            r.print(printstr)
          r.print("Median: Inst SDR=%f, SIR=%f, SAR=%f, Vocal SDR=%f, SIR=%f, SAR=%f" % (np.median(sdr_inst), np.median(sir_inst), np.median(sar_inst), np.median(sdr_vocal), np.median(sir_vocal), np.median(sar_vocal)))

if __name__ == "__main__":
  main()
