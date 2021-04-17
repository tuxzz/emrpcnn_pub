import sys, os, pickle
from mir_util import *
import config as cfg
import librosa
import scipy.signal as sp
sys.path.append("../lib")
import simpleopt
from common import *
import random

n_sample = 0
data_path = cfg.dsd_path
mixed_list = []
inst_list = []
vocal_list = []
n_sample = 0
n_feature = cfg.frame_size // 2
n_aug = simpleopt.get("aug", default=4, ok=int)

print("* Generate spectrograms")
np.random.seed(0x41526941)
random.seed(0x41526941)
for (root, dirs, files) in os.walk(data_path+"/Mixtures/Dev/"):
  for d in sorted(dirs):
    print(d)
    for i in range(n_aug + 1):
      filename_vocal = os.path.join(data_path, "Sources", "Dev", d, "vocals.wav")
      filename_mix = os.path.join(data_path, "Mixtures", "Dev", d, "mixture.wav")
      mixed_wav, sr_orig = loadWav(filename_mix)
      mixed_wav = np.sum(mixed_wav, axis=1)
      mixed_wav = sp.resample_poly(mixed_wav, cfg.sr, sr_orig).astype(np.float32)
      vocals_wav, _ = loadWav(filename_vocal)
      vocals_wav = np.sum(vocals_wav, axis=1)
      vocals_wav = sp.resample_poly(vocals_wav, cfg.sr, sr_orig).astype(np.float32)
      inst_wav = mixed_wav - vocals_wav
      if i != 0:
        vocals_wav = rndshift(vocals_wav, 4000)
        inst_wav = rndshift(inst_wav, 4000)
        vocals_wav *= np.random.uniform(0.5, 1.5)
        mixed_wav = inst_wav + vocals_wav
      mixed_spec = to_spec(mixed_wav)
      mixed_spec_mag = np.abs(mixed_spec)
      vocals_spec = to_spec(vocals_wav)
      vocals_spec_mag = np.abs(vocals_spec)
      inst_spec = to_spec(inst_wav)
      inst_spec_mag = np.abs(inst_spec)
      max_tmp = np.max(mixed_spec_mag)

      mixed_list.append(mixed_spec_mag / max_tmp)
      vocal_list.append(vocals_spec_mag / max_tmp)
      inst_list.append(inst_spec_mag / max_tmp)

      n_sample += 1
print("* Write cache")
assert n_sample > 0
pickle.dump((mixed_list, vocal_list, inst_list, n_sample), open(cfg.dsd2_cache_path, "wb"), 4)

print("* Done")
