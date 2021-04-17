import sys, os
sys.path.append("../lib")
sys.path.append("../srv")
import numpy as np
import pylab as pl
import scipy.interpolate as ipl
import geneop
import simpleopt

breakpoint_path = simpleopt.get("input")
dataset_type = simpleopt.get("dataset")
score_type = simpleopt.get("score")
assert score_type in ("test", "valid")
init_status_path = os.path.join(breakpoint_path, "20_gen_init.pickle")
gen_status_path = os.path.join(breakpoint_path, "30_gen_%04d.pickle")

n_gen = 9999

def do(i_gen, path):
  import evo_g1 as g1
  evo = g1.EvoCore()
  evo.load(path)
  evo.population_genome.sort(key=lambda x:x[1][0] if score_type == "test" else x[2], reverse=True)
  a = "%04d" % (i_gen,) if i_gen != -1 else "init"
  b = "%d" % (i_gen,) if i_gen != -1 else "INIT"
  c = {"mir2": "MIR", "dsd2": "DSD", "mus2": "MUS"}[dataset_type]
  print("v1 g1 %s %d gen_%s_1 S-%s-1-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[0][0]), a, b, c))
  print("v1 g1 %s %d gen_%s_2 S-%s-2-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[1][0]), a, b, c))
  print("v1 g1 %s %d gen_%s_3 S-%s-3-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[3][0]), a, b, c))
i_gen = -1
#do(i_gen, init_status_path)
if dataset_type == "mir2":
  gen_list = (29, 16, 8, 1)
elif dataset_type == "dsd2":
  gen_list = (49, 31, 16, 8, 1)
elif dataset_type == "mus2":
  gen_list = (16, 8, 1)
else:
  assert False
for i_gen in gen_list:
  try:
    do(i_gen, gen_status_path % i_gen)
  except:
    continue
