import sys, os
import numpy as np
import eval_script_reader
sys.path.append("../lib")
import simpleopt

no_eval = simpleopt.get_switch("no_eval")
use_all = simpleopt.get_switch("all")

mach_time_ratio = []
mach_name_list = []
for x in sys.argv[1:]:
  l = [a.strip() for a in x.split("@") if a.strip()]
  if len(l) == 2:
    mach_time_ratio.append(l[0])
    mach_name_list.append(l[1])
n_machine = len(mach_time_ratio)
sdr_time_ratio = 1.5
script_line_list_base = []
time_est_dict = {
  ("mir2", "g1"): 8.0,
  ("dsd2", "g1"): 20.0,
  ("dsd4", "g1"): 20.0,
  ("mir2", "nsga2"): 5.0,
  ("dsd2", "nsga2"): 13.0,
  ("dsd4", "nsga2"): 18.0,
  ("mir2", "any"): 8.0,
  ("dsd2", "any"): 16.0,
  ("dsd4", "any"): 16.0,
}
gpu_time_dict = {
  "2070s": 1.0,
  "1080ti": 0.9,
  "titan_xp": 0.9,
  "titan_rtx": 0.67,
  "titan_v": 0.825,
  "2080ti": 0.725,
  "1080": 1.5,
}

mach_time_ratio = [gpu_time_dict[x] if isinstance(x, str) else x for x in mach_time_ratio]
mach_time_list = [0.0] * n_machine
mach_task_count_list = [0] * n_machine
mach_line_list = [script_line_list_base[:] for _ in range(n_machine)]
conflict_set = set()
for line in eval_script_reader.load_eval_script("eval_script.gen"):
  gene_type, evo_type, eval_type, gene_value = line.gene_type, line.evo_type, line.eval_type, str(line.gene_value)
  alias_name_0 = line.alias_name_list[0]
  if (eval_type, gene_value) in conflict_set:
    print("! Skipped conflict line %d@`%s`" % (line.i_line, line.raw_text,), file=sys.stderr)
    continue
  if not use_all and os.path.exists(os.path.join("./extracted_eval_result", "%s_%s_%s_ex.txt" % (eval_type, gene_type, gene_value))):
    print("! Skipped evaluated line %d@`%s`" % (line.i_line, line.raw_text,), file=sys.stderr)
    continue
  conflict_set.add((eval_type, gene_value))
  task_name = "_".join([gene_type, evo_type, eval_type, gene_value, alias_name_0])
  #print(line)
  assert gene_type in ("v1", "sa", "shn",)
  assert evo_type in ("g1", "nsga2", "any",)
  assert eval_type in ("mir2", "dsd2", "dsd4",)

  i_mach = np.argmin(mach_time_list)
  mach_time_list[i_mach] += time_est_dict[(eval_type, evo_type)] * (sdr_time_ratio if alias_name_0.endswith("_sdr") else 1.0) * mach_time_ratio[i_mach]
  mach_line_list[i_mach].append("# %s" % ("_".join((gene_type, evo_type, eval_type, alias_name_0)),))
  mach_line_list[i_mach].append("if [ ! -f ./mach_status/%s.train.done ]; then" % (task_name,))
  mach_line_list[i_mach].append("  python3 %s_train.py -ver=%s -gene=%s" % (eval_type, gene_type, gene_value,))
  mach_line_list[i_mach].append("  touch ./mach_status/%s.train.done" % (task_name,))
  mach_line_list[i_mach].append("else")
  mach_line_list[i_mach].append("  echo %s is already trained." % (task_name,))
  mach_line_list[i_mach].append("fi")
  if not no_eval:
    mach_line_list[i_mach].append("if [ ! -f ./mach_status/%s.eval.done ]; then" % (task_name,))
    mach_line_list[i_mach].append("  python3 %s_eval.py -ver=%s -gene=%s -step=%d" % (eval_type, gene_type, gene_value, 63000 if eval_type == "mir2" else 630000))
    mach_line_list[i_mach].append("  touch ./mach_status/%s.eval.done" % (task_name,))
    mach_line_list[i_mach].append("else")
    mach_line_list[i_mach].append("  echo %s is already evaluated." % (task_name,))
    mach_line_list[i_mach].append("fi")
  else:
    mach_line_list[i_mach].append("echo Evaluating for %s is skipped." % (task_name,))
  mach_task_count_list[i_mach] += 1

for i_mach in range(n_machine):
  mach_line_list[i_mach] = [
    "#!/bin/sh\n",
    "if [ ! -f ./mir2_cache.pickle ]; then",
    "  python3 mir2_mkcache.py",
    "fi",
    "if [ ! -f ./dsd2_cache.pickle ]; then",
    "  python3 dsd2_mkcache.py",
    "fi",
    "mkdir -p ./mach_status",
    "",
  ] + mach_line_list[i_mach]
  mach_line_list[i_mach].append("")
  mach_line_list[i_mach].append("echo Everything done.")
  print("* Machine %d: to_eval=%d, est_time=%f" % (i_mach, mach_task_count_list[i_mach], mach_time_list[i_mach]))
  print("\n".join(mach_line_list[i_mach]), file=open("do_%s.sh" % (mach_name_list[i_mach],), "w"))
print("* Total: to_eval=%d, est_time=%f" % (sum(mach_task_count_list), max(mach_time_list)))
