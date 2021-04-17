import os, sys, re
import count_var
import numpy as np
sys.path.append("../lib")
import simpleopt

out_dir = "./extracted_eval_result"

re_mir2_full = re.compile(r"GNSDR\s*\[\s*Accompaniments,\s*voice\s*\]\s*\n\s*\[\s*([0-9\.\-e]+)\s*([0-9\.\-e]+)\s*\]\s*\n\s*GSIR \[\s*Accompaniments,\s*voice\s*\]\s*\n\s*\[\s*([0-9\.\-e]+)\s*([0-9\.\-e]+)\s*\]\s*\n\s*GSAR\s*\[\s*Accompaniments, voice\s*\]\s*\n\s*\[\s*([0-9\.\-e]+)\s*([0-9\.\-e]+)\s*\]")
re_dsd2_mean_sdr = re.compile(r"Median:\s*Inst\s*SDR=([0-9\.\-e]+),\s*SIR=([0-9\.\-e]+),\s*SAR=([0-9\.\-e]+),\s*Vocal\s*SDR=([0-9\.\-e]+),\s*SIR=([0-9\.\-e]+),\s*SAR=([0-9\.\-e]+)")

use_step_list = [int(x) for x in simpleopt.get_multi("step")]

print("* Load")
cand_list = {}
for root, dir_list, file_list in os.walk("./eval_output"):
  for filename in file_list:
    path = os.path.join(root, filename)
    print(path)
    assert filename[-4:] == ".txt"
    mark_list = filename[:-4].split("_")
    assert len(mark_list) == 4
    assert mark_list[3][:4] == "step"
    mark_list[2] = int(mark_list[2])
    mark_list[3] = int(mark_list[3][4:])
    line_list = [x.strip("\n") for x in open(path, "r").readlines()]
    ok = False
    if mark_list[0] == "mir2":
      g = re_mir2_full.search("\n".join(line_list))
      if g:
        sdr_inst, sdr_vocal = float(g.group(1)), float(g.group(2))
        sir_inst, sir_vocal = float(g.group(3)), float(g.group(4))
        sar_inst, sar_vocal = float(g.group(5)), float(g.group(6))
        mean_sdr = np.mean((sdr_inst, sdr_vocal))
        mean_sir = np.mean((sir_inst, sir_vocal))
        mean_sar = np.mean((sar_inst, sar_vocal))
        ok = True
    elif mark_list[0] == "dsd2":
      for line in line_list:
        g = re_dsd2_mean_sdr.match(line)
        if g:
          sdr_inst, sdr_vocal = float(g.group(1)), float(g.group(4))
          sir_inst, sir_vocal = float(g.group(2)), float(g.group(5))
          sar_inst, sar_vocal = float(g.group(3)), float(g.group(6))
          mean_sdr = np.mean((sdr_inst, sdr_vocal))
          mean_sir = np.mean((sir_inst, sir_vocal))
          mean_sar = np.mean((sar_inst, sar_vocal))
          ok = True
          break
    else:
      assert False
    assert ok, path
    cand_key = tuple(mark_list[:3])
    l = cand_list.get(cand_key, None)
    if l is None:
      l = []
      cand_list[cand_key] = l
    l.append((mark_list, mean_sdr, mean_sir, mean_sar, line_list))
    print("mean_sdr=%f, mean_sir=%f, mena_sar=%f" % (mean_sdr, mean_sir, mean_sar))

print("* Compare")
os.makedirs(out_dir, exist_ok=True)
old_argv = sys.argv
h_max = {}
h_min = {}
def adddefault(d, tp, k):
  v = d.get(tp, None)
  if v is None:
    v = {}
    d[tp] = v
  v[k] = v.get(k, 0) + 1
def dsrt(d):
  return {k: sorted(v.items(), key=lambda x:x[1]) for k, v in d.items()}
for k, v in cand_list.items():
  v.sort(key=lambda x:x[1])
  adddefault(h_min, v[0][0][0], v[0][0][-1])
  adddefault(h_max, v[-1][0][0], v[-1][0][-1])
  if use_step_list:
    v = [x for x in v if x[0][-1] in use_step_list]
  print(k)
  mark_list, mean_sdr, mean_sir, mean_sar, line_list = v[-1]
  mark_list = [str(x) for x in mark_list]
  n_param, n_forward_flops = count_var.count(mark_list[1], int(mark_list[2]), 2)
  print(" : best_step=%s, mean_sdr=%f, mean_sir=%f, mean_sar=%f, n_param=%d, n_forward_flops=%d" % (mark_list[3], mean_sdr, mean_sir, mean_sar, n_param, n_forward_flops))
  pre_list = [
    "# Extracted Result",
    ": gene_type=%s, gene_value=%s" % (mark_list[1], mark_list[2]),
    ": eval_type=%s, step=%s, mean_sdr=%f, mean_sir=%f, mean_sar=%f, n_param=%d, n_forward_flops=%d" % (mark_list[0], mark_list[3], mean_sdr, mean_sir, mean_sar, n_param, n_forward_flops),
  ]
  s = "\n".join(pre_list + line_list)
  with open(os.path.join(out_dir, "_".join(mark_list[:3] + ["ex.txt"])), "wb") as f:
    f.write(s.encode("utf-8"))
print("h_min: %r" % (dsrt(h_min),))
print("h_max: %r" % (dsrt(h_max),))
