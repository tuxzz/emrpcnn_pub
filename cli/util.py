
'''
Part of the code modified from
https://github.com/andabi/music-source-separation
'''

import numpy as np
from mir_eval.separation import bss_eval_sources

def calc_sdr(w_real, w_pred):
  n_real, = w_real.shape
  n_pred, = w_pred.shape
  n = min(n_real, n_pred)
  
  w_real, w_pred = w_real[:n], w_pred[:n]

  sdr, _, _, _ = bss_eval_sources(w_real, w_pred, compute_permutation=True)
  return sdr

def bss_eval(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len = pred_src1_wav.shape[0]
    src1_wav = src1_wav[:len]
    src2_wav = src2_wav[:len]
    mixed_wav = mixed_wav[:len]
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), compute_permutation=True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), compute_permutation=True)
    nsdr = sdr - sdr_mixed
    return nsdr, sir, sar, len

def bss_eval_sdr(src1_wav, pred_src1_wav):
        len_cropped = pred_src1_wav.shape[0]
        src1_wav = src1_wav[:len_cropped]

        sdr, _, _, _ = bss_eval_sources(src1_wav,
                                            pred_src1_wav, compute_permutation=True)
        return sdr
