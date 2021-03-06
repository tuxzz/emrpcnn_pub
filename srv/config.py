##### ⚠️ Do not change anything that said "do not change" unless you know what you are doing ⚠️ #####
import sys
sys.path.append("../lib")
import os
import geneop

# Section: Server
#  Iterable[str]: Only client that version in this list can be accepted by server
allowed_cli_version = ("191007r0",)
#  str|None: IP address to listen, None means Flask default (usually localhost).
listen_addr = "0.0.0.0"
#  str|None: TCP port to listen, None means Flask default (usually 5000).
listen_port = None
#  str|None: Worker result cache path, None means disabled.
worker_cache_path = "./worker_cache.pickle"
#  str|None: Worker result cache backup path, None means disabled. 
worker_cache_backup_path = worker_cache_path + ".bak"

# Section: Evolution
#  int: Maximum number of generations to calculate.
#       Server will be stopped if this value is met.
# n_gen = 300 # ⚠️ defined dynmatically in the end of this config ⚠️
#  str: Algorithm of evolution to use.
#       "g1" for single target, "nsga2" for two target.
evo_type = "nsga2"
#  module: Python module for specified evo_type (do not change!)
evo_module = __import__("evo_%s" % (evo_type,)) 
#  float: Ratio of pure random gene in initial population.
#         0.0 means no pure random gene, 1.0 means all population are built with pure random gene.
init_random_ratio = 0.0
p2 = 0.02

# Section: Gene
#  str: "v1" for the paper (do not change!)
gene_type = "v1"
geneop.load_type(gene_type)
#  List[List[bool]]: Handcrafted initial population (do not change!)
if init_random_ratio != 1.0:
  if gene_type == "v1":
    manual_gene_list = [
      geneop.cvtintlst(x, 142) for x in [
        4182591019167972528534244115322478782824676,
      ]
    ]
  else:
    assert False

# Section: Evolution(Additional)
#  int: Maximum number of bits to flip when generate gene from handcrafted popluation (do not change!)
if init_random_ratio != 1.0:
  if gene_type in ("v1",):
    gene_jitter_count = 20
  else:
    assert False

# Section: Worker
#  str: Specify the dataset used for evaluating population.
#       Can be one of "mir2", "dsd2", "mus2"
worker_type = "mus2"

#  int: Specify how many results need to be returned from the worker (do not change!)
if evo_type == "g1":
  n_result = 1
elif evo_type == "nsga2":
  n_result = 2

#  dict: Configuration of training and evaluating parameters for worker client
if worker_type == "toy":
  worker_config = {
    "n_result": n_result
  }
else: # mir2, dsd2, dsd4
  def boundary_00():
    result_format = ("sdr", "neg_mega_pc", "neg_gflops", "valid_sdr")
    valid_idx = 3
    if evo_type == "g1":
      result_needed = (0,)
    elif evo_type == "nsga2":
      result_needed = (0, 1,)

    if worker_type == "mir2":
      n_step = 1500
      n_train = 100
      n_eval = 55
      n_valid = 20
    elif worker_type == "dsd2":
      n_step = 3100
      n_train = 30
      n_eval = 15
      n_valid = 5
    elif worker_type == "mus2":
      n_step = 15000
      n_train = 60
      n_eval = 30
      n_valid = 10

    return {
      "result_format": result_format,
      "batch_size": 3,
      "max_lr": 3e-4,
      "min_lr": 1e-4,
      "warmup_fac": 0.3,
      "first_lr_period": 1000,
      "warmup_period": 1000,
      "n_hop_per_sample": 64,
      "work_sr": 16000,
      "hop_size": 512,
      "fft_size": 2048,
      "n_feature": 1024,
      "n_step": n_step,
      "n_train": n_train,
      "n_eval": n_eval,
      "n_valid": n_valid,
    }, result_needed, valid_idx
  worker_config, result_needed, valid_idx = boundary_00()
  del boundary_00

# Section: Evolution checkpoint
breakpoint_path = "./%s_%s_%s_%.1f" % (gene_type, evo_type, worker_type, init_random_ratio,)
init_status_path = os.path.join(breakpoint_path, "20_gen_init.pickle")
gen_status_path = os.path.join(breakpoint_path, "30_gen_%04d.pickle")
status_path = os.path.join(breakpoint_path, "00_status.pickle")

# Section: Evolution, pass 2
if evo_type == "g1":
  n_gen = 30
else:
  n_gen = 100

# Section: Check
assert evo_type in ("g1", "nsga2"), "bad evo_type"
assert worker_type in ("mir2", "dsd2", "mus2"), "bad worker_type"
if evo_module.EvoCore.n_min_result is not None:
  assert n_result >= evo_module.EvoCore.n_min_result, "bad n_result"
if evo_module.EvoCore.n_max_result is not None:
  assert n_result <= evo_module.EvoCore.n_max_result, "bad n_result"
if worker_type != "toy":
  assert all(x in ("sdr", "neg_mega_pc", "neg_gflops", "valid_sdr") for x in worker_config["result_format"]), "bad result_format"
assert 0.0 <= init_random_ratio <= 1.0, "bad init_random_ratio"
os.makedirs(breakpoint_path, exist_ok=True)
