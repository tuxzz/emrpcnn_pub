import sys, os, pickle
sys.path.append("../lib")
import simpleopt
from common import *
from mir_util import *
import config as cfg
import random

data_path = cfg.mir_wav_path
mixed_list = []
inst_list = []
vocal_list = []
n_sample = 0
n_feature = cfg.frame_size // 2
n_aug = simpleopt.get("aug", default=4, ok=int)

print("* Generate spectrogram")
np.random.seed(0x41526941)
random.seed(0x41526941)
for (root, dirs, files) in os.walk(data_path):
  for filename in [x for x in sorted(files) if x.startswith("abjones") or x.startswith("amy")]:
    print(filename)
    path = os.path.join(root, filename)
    for i in range(n_aug + 1):
      w, sr = loadWav(path)
      if sr != cfg.sr:
        w = sp.resample_poly(w, cfg.sr, sr).astype(np.float32)
        sr = cfg.sr
      mixed_wav = np.sum(w, axis=1)
      inst_wav = w[:, 0]
      vocals_wav = w[:, 1]
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
pickle.dump((mixed_list, vocal_list, inst_list, n_sample), open(cfg.mir2_cache_path, "wb"), 4)

print("* Done")
