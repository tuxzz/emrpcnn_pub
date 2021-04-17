import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os, sys
sys.path.append("../lib")

import simpleopt

sr = 8000
frame_size = 1024
hop_size = 256

ckpt_root_path = "./ckpt"

dsd_path = "../DSD100"
dsd2_cache_path = "dsd2_cache.pickle"

mir_wav_path = "../MIR-1K/Wavfile"
mir2_cache_path = "mir2_cache.pickle"

mus_root_path = "../musdb18_decoded/"
mus_train_path = "../musdb18_decoded/train"
mus_eval_path = "../musdb18_decoded/test"

sess_cfg = tf.compat.v1.ConfigProto(
  gpu_options=tf.compat.v1.GPUOptions(
    allow_growth=True,
    per_process_gpu_memory_fraction=1.0,
  ),
  allow_soft_placement=True,
)

gene_ver = None
gene_value = None
DSD2Config = None
MIR2Config = None
MUS2FConfig = None

def load_config(gene_ver=None, gene_value=None):
  import config as cfg
  if not os.path.exists(ckpt_root_path):
    os.makedirs(ckpt_root_path, exist_ok=True)
  if gene_ver is None:
    gene_ver = simpleopt.get("ver")
  if gene_value is None:
    gene_value = int(simpleopt.get("gene"))
  if gene_value is None:
    gene_value = int(simpleopt.get("gene"))

  class DSD2Config:
    ckpt_name = "dsd2_%s_%d" % (gene_ver, gene_value,)
    batch_size = 1 if gene_ver in ("sa",) else 3
    ckpt_path = os.path.join(ckpt_root_path, ckpt_name)
    max_lr = 1e-4 if gene_ver in ("sa", "sa2",) and gene_value in (4,) else 3e-4
    min_lr = 1e-5
    first_decay_period = 10000 * (3 // batch_size)
    final_step = 630001 * (3 // batch_size)
    ckpt_step = 10000 * (3 // batch_size)

  class MIR2Config:
    ckpt_name = "mir2_%s_%d" % (gene_ver, gene_value,)
    batch_size = 1 if gene_ver in ("sa",) else 3
    ckpt_path = os.path.join(ckpt_root_path, ckpt_name)
    max_lr = 3e-4
    min_lr = 1e-5
    first_decay_period = 1000 * (3 // batch_size)
    final_step = 63001 * (3 // batch_size)
    ckpt_step = 1000 * (3 // batch_size)

  class MUS2FConfig:
    ckpt_name = "mus2f_%s_%d" % (gene_ver, gene_value,)
    batch_size = 1 if gene_ver in ("sa",) else 3
    ckpt_path = os.path.join(ckpt_root_path, ckpt_name)
    max_lr = 1e-4 if gene_ver in ("sa", "sa2",) and gene_value in (4,) else 2e-4 # FFFFFFFFFF
    min_lr = 1e-5
    first_decay_period = 10000 * (3 // batch_size)
    final_step = 630001 * (3 // batch_size)
    ckpt_step = 10000 * (3 // batch_size)

  cfg.gene_ver = gene_ver
  cfg.gene_value = gene_value
  cfg.DSD2Config = DSD2Config
  cfg.MIR2Config = MIR2Config
  cfg.MUS2FConfig = MUS2FConfig

class ConfigBoundary:
  def __init__(self, gene_ver=None, gene_value=None):
    import config as cfg
    self.gene_ver = cfg.gene_ver
    self.gene_value = cfg.gene_value
    self.DSD2Config = cfg.DSD2Config
    self.MIR2Config = cfg.MIR2Config
    self.MUS2FConfig = cfg.MUS2FConfig
    self.new_gene_ver = gene_ver
    self.new_gene_value = gene_value

  def __enter__(self):
    load_config(self.new_gene_ver, self.new_gene_value)
    return self
  
  def __exit__(self, tp, value, trace):
    import config as cfg
    cfg.gene_ver = self.gene_ver
    cfg.gene_value = self.gene_value
    cfg.DSD2Config = self.DSD2Config
    cfg.MIR2Config = self.MIR2Config
    cfg.MUS2FConfig = self.MUS2FConfig
