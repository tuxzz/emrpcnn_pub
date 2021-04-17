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

max_sdr_list = []
mean_sdr_list = []
sdr_25_list = []
sdr_50_list = []
sdr_75_list = []

min_param_list = []
mean_param_list = []
param_25_list = []
param_50_list = []
param_75_list = []
def do(path):
  import evo_nsga2 as nsga2
  evo = nsga2.EvoCore()
  evo.load(path)
  sdr_list = sorted([score[0] if score_type == "test" else v for gene, score, v in evo.population_genome])
  param_list = [-score[1] for gene, score, v in evo.population_genome]
  sort_param_idx_list = np.argsort(param_list)
  param_list = sorted(param_list)

  max_sdr_list.append(np.max(sdr_list))
  mean_sdr_list.append(np.mean(sdr_list))
  sdr_25, sdr_50, sdr_75 = ipl.interp1d(np.linspace(0.0, 1.0, len(sdr_list)), sdr_list, kind="linear", bounds_error=True)([0.25, 0.5, 0.75])
  sdr_25_list.append(sdr_25)
  sdr_50_list.append(sdr_50)
  sdr_75_list.append(sdr_75)

  min_param_list.append(np.min(param_list))
  mean_param_list.append(np.mean(param_list))
  param_25, param_50, param_75 = ipl.interp1d(np.linspace(0.0, 1.0, len(param_list)), param_list, kind="linear", bounds_error=True)([0.25, 0.5, 0.75])
  param_25_list.append(param_25)
  param_50_list.append(param_50)
  param_75_list.append(param_75)
i_gen = -1
do(init_status_path)
for i_gen in range(n_gen):
  try:
    do(gen_status_path % i_gen)
  except:
    continue

rec_p25, rec_p50, rec_p75 = np.median(param_25_list), np.median(param_50_list), np.median(param_75_list)

p25_sdr_list = []
p33_sdr_list = []
p50_sdr_list = []
p66_sdr_list = []
p75_sdr_list = []
p90_sdr_list = []

def do(i_gen, path):
  import evo_nsga2 as nsga2
  evo = nsga2.EvoCore()
  evo.load(path)
  sdr_list = np.asarray([score[0] if score_type == "test" else v for gene, score, v in evo.population_genome])
  eval_sdr_list = np.asarray([score[0] for gene, score, v in evo.population_genome])
  param_list = np.asarray([-score[1] for gene, score, v in evo.population_genome])
  
  n = len(param_list)
  idx_list = np.argsort(param_list)

  i_fast = np.argmin(param_list)
  '''i_p25 = np.argmin(np.abs(param_list - rec_p25))
  i_p50 = np.argmin(np.abs(param_list - rec_p50))
  i_p75 = np.argmin(np.abs(param_list - rec_p75))'''
  i_p25 = idx_list[n // 4]
  i_p33 = idx_list[n // 3]
  i_p50 = idx_list[n // 2]
  i_p66 = idx_list[n * 2 // 3]
  i_p75 = idx_list[n * 3 // 4]
  i_p90 = idx_list[n * 9 // 10]
  i_sdr = np.argmax(sdr_list)
  i_eval_sdr = np.argmax(eval_sdr_list)

  p25_sdr_list.append(sdr_list[i_p25])
  p33_sdr_list.append(sdr_list[i_p33])
  p50_sdr_list.append(sdr_list[i_p50])
  p66_sdr_list.append(sdr_list[i_p66])
  p75_sdr_list.append(sdr_list[i_p75])
  p90_sdr_list.append(sdr_list[i_p90])

  if i_gen in (99, 50, 25, 1):
    a = "%04d" % (i_gen,) if i_gen != -1 else "init"
    b = "%d" % (i_gen,) if i_gen != -1 else "INIT"
    c = {"mir2": "MIR", "dsd2": "DSD", "mus2": "MUS"}[dataset_type]
    if i_gen == 99 or (i_gen == 50 and dataset_type == "mus2"):
      print("v1 nsga2 %s %d gen_%s_fast M-%s-1-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_fast][0]), a, b, c))
      print("v1 nsga2 %s %d gen_%s_p25 M-%s-2-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p25][0]), a, b, c))
      print("v1 nsga2 %s %d gen_%s_p33 M-%s-3-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p33][0]), a, b, c))
      print("v1 nsga2 %s %d gen_%s_p50 M-%s-4-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p50][0]), a, b, c))
      print("v1 nsga2 %s %d gen_%s_p66 M-%s-5-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p66][0]), a, b, c))
      print("v1 nsga2 %s %d gen_%s_p75 M-%s-6-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p75][0]), a, b, c))
      print("v1 nsga2 %s %d gen_%s_p90 M-%s-7-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p90][0]), a, b, c))
      print("v1 nsga2 %s %d gen_%s_sdr M-%s-8-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_sdr][0]), a, b, c))
      #print("v1 nsga2 %s %d gen_%s_eval_sdr M-%s-X-%s" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_eval_sdr][0]), a, b, c))
      print("")
    else:
      print("v1 nsga2 %s %d gen_%s_fast" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_fast][0]), a))
      print("v1 nsga2 %s %d gen_%s_p25" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p25][0]), a))
      print("v1 nsga2 %s %d gen_%s_p33" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p33][0]), a))
      print("v1 nsga2 %s %d gen_%s_p50" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p50][0]), a))
      print("v1 nsga2 %s %d gen_%s_p66" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p66][0]), a))
      print("v1 nsga2 %s %d gen_%s_p75" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p75][0]), a))
      print("v1 nsga2 %s %d gen_%s_p90" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_p90][0]), a))
      print("v1 nsga2 %s %d gen_%s_sdr" % (dataset_type, geneop.cvtlstint(evo.population_genome[i_sdr][0]), a))
      print("")
do(-1, init_status_path)
for i_gen in range(n_gen):
  try:
    do(i_gen, gen_status_path % i_gen)
  except:
    continue

print("! NSGA2 param count: 25%%=%f, 50%%=%f, 75%%=%f" % (rec_p25, rec_p50, rec_p75,))

x = np.arange(-1, len(max_sdr_list) - 1)
pl.figure()
pl.title("SDR")
pl.plot(x, max_sdr_list, label="Max")
#pl.plot(x, sdr_25_list, label="25")
#pl.plot(x, sdr_50_list, label="50")
#pl.plot(x, sdr_75_list, label="75")
pl.plot(x, mean_sdr_list, label="Mean")
pl.plot(x, p25_sdr_list, label="p25")
pl.plot(x, p50_sdr_list, label="p50")
pl.plot(x, p75_sdr_list, label="p75")
pl.legend()

pl.figure()
pl.title("Param")
pl.plot(x, min_param_list, label="0")
pl.plot(x, param_25_list, label="25")
pl.plot(x, param_50_list, label="50")
pl.plot(x, param_75_list, label="75")
pl.plot(x, mean_param_list, label="Mean")
pl.hlines(rec_p25, x[0], x[-1], label="rec_25")
pl.hlines(rec_p50, x[0], x[-1], label="rec_50")
pl.hlines(rec_p75, x[0], x[-1], label="rec_75")
pl.legend()
pl.show()
