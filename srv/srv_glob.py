import threading, master, queue, time
from typing import *

time_to_alive = 60.0

glob_lock = threading.RLock()
status_strbuffer = []
online_client_dict = {}
task_input_queue = queue.Queue(1)
task_output_queue = queue.Queue(1)
task_temp = []
master_obj: master.Master = None
evo_obj = None

incoming_population = []
prev_population = []

def print_status(category: str, value: str, *args, **kwargs):
  import sys
  if "file" in kwargs:
    raise ValueError("Keyword argument `file` is not supported in print_status")
  else:
    import io
    f = io.StringIO()
    print("[%s]%s" % (category, value), *args, **kwargs, file=f)
    s = f.getvalue()
    status_strbuffer.append((category, s))
    print(s, end="")

def clear_zomibes(t: float):
  import srv_glob as glob
  old_cli = glob.online_client_dict
  glob.online_client_dict = {cli_id: cli for cli_id, cli in online_client_dict.items() if t - cli["last_seen"] < time_to_alive}
  for k, v in old_cli.items():
    if k not in glob.online_client_dict:
      print_status("SRV", "Removed zombie client #%d(%s)" % (k, v["hostname"]))
  for x in task_temp:
    if not (x["cli_id"] is None or x["cli_id"] in glob.online_client_dict):
      print_status("SRV", "Unassigned task for zombie client %d" % (x["cli_id"],))
      x["cli_id"] = None

def unassign_task(task_id: int):
  task = task_temp[task_id]
  assert task["result"] is None
  assert task["cli_id"] is not None
  task["cli_id"] = None

def get_task(cli_id: int):
  if task_temp == []:
    try:
      q = task_input_queue.get_nowait()
    except queue.Empty:
      return "wait", None
    for x in q:
      task_temp.append({
        "gene": x,
        "cli_id": None,
        "result": None,
        "start_time": None,
      })
  for i, x in enumerate(task_temp):
    if x["result"] is None and x["cli_id"] is None:
      x["cli_id"] = cli_id
      x["start_time"] = time.time()
      return "run", (i, x["gene"])
  return "wait", None

def set_result(task_id: int, result: List[float]):
  import config as cfg
  task = task_temp[task_id]
  assert task["result"] is None
  assert task["cli_id"] is not None
  assert isinstance(result, (List, Tuple))
  hostname = "Unknown hostname"
  cli = online_client_dict.get(task["cli_id"], None)
  if cli is not None:
    hostname = cli["hostname"]
  t = time.time() - task["start_time"]
  result_str = ", ".join(["%s=%r" % (x, y) for x, y in zip(cfg.worker_config["result_format"], result)])
  print_status("EVAL", "%s, time_usage=%.2fs (client #%d:`%s`)" % (result_str, t, task["cli_id"], hostname))
  task["cli_id"] = None
  task["start_time"] = None
  task["result"] = result
  if all(x["result"] is not None for x in task_temp):
    task_output_queue.put([x["result"] for x in task_temp])
    task_temp.clear()
  else:
    incoming_population.append(result)
    create_plot()

def create_plot():
  import pylab as pl
  import config as cfg
  if cfg.n_result == 2:
    pl.xlabel("Target 0")
    pl.ylabel("Validation Score")
    if evo_obj.population_genome and all(valid_score for _, _, valid_score in evo_obj.population_genome):
      l = list(zip(*[(valid_score, score[1]) for _, score, valid_score in evo_obj.population_genome]))
      pl.scatter(l[0], l[1])
    elif prev_population:
      l = list(zip(*prev_population))
      pl.scatter(l[0], l[1])
    if incoming_population:
      l = list(zip(*((x[cfg.valid_idx], x[1],) for x in incoming_population)))
      pl.scatter(l[0], l[1])
  elif cfg.n_result == 1:
    pl.xlabel("")
    pl.ylabel("Validation Score")
    if evo_obj.population_genome and all(valid_score for _, _, valid_score in evo_obj.population_genome):
      l = [valid_score for _, _, valid_score in evo_obj.population_genome]
      pl.scatter([0] * len(l), l)
    elif prev_population:
      l = [x[0] for x in prev_population]
      pl.scatter([0] * len(l), l)
    if incoming_population:
      l = [x[cfg.valid_idx] for x in incoming_population]
      pl.scatter([0] * len(l), l)
  pl.savefig("chart.svg")
  pl.close()