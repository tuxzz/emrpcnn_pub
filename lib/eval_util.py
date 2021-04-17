def bss_eval(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
  import numpy as np
  from mir_eval.separation import bss_eval_sources
  n = pred_src1_wav.shape[0]
  src1_wav = src1_wav[:n]
  src2_wav = src2_wav[:n]
  mixed_wav = mixed_wav[:n]
  sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]), np.array([pred_src1_wav, pred_src2_wav]), compute_permutation=True)
  sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]), np.array([mixed_wav, mixed_wav]), compute_permutation=True)
  # sdr, sir, sar, _ = bss_eval_sources(src2_wav,pred_src2_wav, False)
  # sdr_mixed, _, _, _ = bss_eval_sources(src2_wav,mixed_wav, False)
  nsdr = sdr - sdr_mixed
  return nsdr, sir, sar, n

def bss_eval_sdr(src_list, pred_src_list):
  from mir_eval.separation import bss_eval_sources
  len_cropped = pred_src_list.shape[-1]
  src_list = src_list[:, :len_cropped]

  sdr, sir, sar, _ = bss_eval_sources(src_list, pred_src_list, compute_permutation=True)
  return sdr, sir, sar

def bss_eval_sdr_framewise(src_list, pred_src_list):
  from mir_eval.separation import bss_eval_sources_framewise
  len_cropped = pred_src_list.shape[-1]
  src_list = src_list[:, :len_cropped]

  sdr, sir, sar, _ = bss_eval_sources_framewise(src_list, pred_src_list, window=44100, hop=44100, compute_permutation=True)
  sdr = [remove_abnormal(x) for x in sdr]
  sir = [remove_abnormal(x) for x in sir]
  sar = [remove_abnormal(x) for x in sar]
  return sdr, sir, sar

def bss_eval_sdr_framewise_2s(src_list, pred_src_list):
  from mir_eval.separation import bss_eval_sources_framewise
  len_cropped = pred_src_list.shape[-1]
  src_list = src_list[:, :len_cropped]

  sdr, sir, sar, _ = bss_eval_sources_framewise(src_list, pred_src_list, window=44100 * 2, hop=44100 * 2, compute_permutation=True)
  sdr = [remove_abnormal(x) for x in sdr]
  sir = [remove_abnormal(x) for x in sir]
  sar = [remove_abnormal(x) for x in sar]
  return sdr, sir, sar

def remove_abnormal(l):
  import numpy as np
  return [x for x in l if np.isfinite(x)]

def bss_eval_sdr_v4(src_list, pred_src_list, do_remove_abnormal=True):
  import museval
  len_cropped = pred_src_list.shape[-1]
  src_list = src_list[:, :len_cropped]
  sdr, isr, sir, sar = museval.evaluate(src_list, pred_src_list, win=44100, hop=44100, mode="v4", padding=True)
  if do_remove_abnormal:
    sdr = [remove_abnormal(x) for x in sdr]
    isr = [remove_abnormal(x) for x in isr]
    sir = [remove_abnormal(x) for x in sir]
    sar = [remove_abnormal(x) for x in sar]
  return sdr, isr, sir, sar

def bss_eval_sdr_v4_2s(src_list, pred_src_list, do_remove_abnormal=True):
  import museval
  len_cropped = pred_src_list.shape[-1]
  src_list = src_list[:, :len_cropped]
  sdr, isr, sir, sar = museval.evaluate(src_list, pred_src_list, win=44100 * 2, hop=44100 * 2, mode="v4", padding=True)
  if do_remove_abnormal:
    sdr = [remove_abnormal(x) for x in sdr]
    isr = [remove_abnormal(x) for x in isr]
    sir = [remove_abnormal(x) for x in sir]
    sar = [remove_abnormal(x) for x in sar]
  return sdr, isr, sir, sar
