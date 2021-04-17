import sys
sys.path.append("../lib")
import geneop
import config as cfg
from filelock import FileLock

def pre_fn():
  import simpleopt
  dataset_type = simpleopt.get("dataset")
  gene_type = simpleopt.get("ver")
  geneop.load_type(gene_type)
  cfg.n = {
    "v1": 142,
    "v1pe": 142,
    "v1f16": 142,
  }[gene_type]
  if dataset_type == "mus2":
    '''cfg.worker_config = {
      "result_format": ("sdr", "neg_gflops", "neg_mega_pc", "valid_sdr"),
      "batch_size": 3,
      "max_lr": 3e-4,
      "min_lr": 1e-4,
      "warmup_fac": 0.3,
      "first_lr_period": 10000,
      "warmup_period": 1000,
      "n_hop_per_sample": 64,
      "work_sr": 16000,
      "hop_size": 512,
      "fft_size": 2048,
      "n_feature": 1024,
    }'''
    cfg.worker_config = {
      "result_format": ("sdr", "neg_gflops", "neg_mega_pc", "valid_sdr"),
      "batch_size": 3,
      "max_lr": 3e-4,
      "min_lr": 1e-4,
      "warmup_fac": 0.3,
      "first_lr_period": 100,
      "warmup_period": 100,
      "n_hop_per_sample": 64,
      "work_sr": 16000,
      "hop_size": 512,
      "fft_size": 2048,
      "n_feature": 1024,
    }
  else:
    cfg.worker_config = {
      "result_format": ("sdr", "neg_gflops", "neg_mega_pc", "valid_sdr"),
      "batch_size": 2,
      "max_lr": 3e-4,
      "min_lr": 1e-4,
      "warmup_fac": 0.3,
      "first_lr_period": 100,
      "warmup_period": 100,
      "n_hop_per_sample": 64,
      "work_sr": 8000,
      "hop_size": 256,
      "fft_size": 1024,
      "n_feature": 512,
    }
  if dataset_type == "mir2":
    cfg.worker_config.update(
      n_train=100,
      n_step=1500,
      n_eval=55,
      n_valid=20,
    )
  elif dataset_type == "dsd2":
    cfg.worker_config.update(
      n_train=30,
      n_step=3100,
      n_eval=15,
      n_valid=5,
    )
  elif dataset_type == "mus2":
    '''cfg.worker_config.update(
      n_train=60,
      n_step=30000,
      n_eval=30,
      n_valid=10,
    )'''
    cfg.worker_config.update(
      n_train=60,
      n_step=3100,
      n_eval=30,
      n_valid=10,
    )
  cfg.worker = __import__("worker_%s" % (dataset_type,))
  cfg.worker.clean_env()
  cfg.worker.pre_fn()

def eval_single(gene):
  gene = geneop.cvtintlst(int(gene), cfg.n)
  return cfg.worker.eval_fn(gene)

def clean_fn():
  del cfg.worker_config
  cfg.worker.clean_env()
  del cfg.worker

def create_lock():
  import simpleopt
  lockpath = simpleopt.get("lockpath")
  if not cfg.lock:
    cfg.lock = FileLock(lockpath)
  
def main():
  import sys
  sys.path.append("../lib")
  import time
  import config as cfg
  import multiprocessing as mp
  import simpleopt
  create_lock()
  dataset_type = simpleopt.get("dataset")
  with mp.Pool(processes=6) as pool:
    cfg.pool = pool
    pre_fn()
    l = []
    for gene in sys.argv[4:]:
      try:
        print("* Acquire GPU...")
        cfg.lock.acquire()
        print("* Got GPU!")
        t = time.time()
        r = eval_single(gene)
        t = time.time() - t
        l.append((gene, *r, t))
      finally:
        cfg.lock.release()
    clean_fn()
    for i, (gene, sdr, neg_gflops, neg_mega_pc, valid_sdr, t) in enumerate(l):
      print("[%d/%d] gene=%s, sdr=%f, neg_gflops=%f, neg_mega_pc=%f, valid_sdr=%f, time=%f" % (i, len(l), gene, sdr, neg_gflops, neg_mega_pc, valid_sdr, t))

if __name__ == "__main__":
  main()
