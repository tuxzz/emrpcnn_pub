import sys
sys.path.append("../lib")
import multiprocessing as mp
import watchdog, config as cfg, requests
from requests.compat import urljoin
import traceback
from filelock import FileLock

class Boundary:
  def __enter__(self):
    print("Boundary") 
    pass
  
  def __exit__(self, tp, value, trace):
    import os
    if cfg.dog is not None:
      cfg.dog.request_stop = True
    url_rpc = urljoin(cfg.srv_url, "/rpc/%d" % (cfg.cli_id,))
    cfg.net_sess.put(url_rpc, json={"cmd": "bye"}, timeout=8.0)
    pass

def send_rpc(cmd: str, *args):
  import os
  url_rpc = urljoin(cfg.srv_url, "/rpc/%d" % (cfg.cli_id,))
  ret = cfg.net_sess.put(url_rpc, json={"cmd": cmd, "args": args}, timeout=8.0)
  if ret.status_code != 200:
    raise RuntimeError("Cannot send rpc command `%s`: HTTP %d(%s)" % (cmd, ret.status_code, ret.text))
  if ret.text != "Success":
    ret = ret.json()
    return ret
  return None

def create_lock():
  import simpleopt
  lockpath = simpleopt.get("lockpath")
  if not cfg.lock:
    cfg.lock = FileLock(lockpath)

def check_rng():
  import random
  import numpy as np
  random.seed(0x41526941)
  np.random.seed(0x41526941)
  l = [1, 2, 3, 4, 5]
  random.shuffle(l)
  assert l == [2, 3, 5, 1, 4], "invalid rng"
  assert (np.random.permutation(5) == [0, 1, 3, 4, 2]).all(), "invalid rng"

def check_framework():
  #return # ⚠️WARNING⚠️
  import worker_mir2, gc
  import numpy as np
  try:
    print("* Acquire GPU...")
    cfg.lock.acquire()
    print("* Got GPU!")
    result = worker_mir2.oneshot("v1", 795070747881445470762713953841528734498944)
  finally:
    cfg.lock.release()
  assert abs(result[0] - 8.87) < 0.25, "Invalid SDR result %f" % (result[0],)
  assert abs(result[1] - -11.612382) < 2e-2, "Invalid FLOPs %f" % (result[1],)
  assert abs(result[2] - -0.095170) < 1e-3, "Invalid param count %f" % (result[2],)
  gc.collect()

def main(pool):
  import requests, time, os, socket
  from urllib3.util.retry import Retry
  from requests.adapters import HTTPAdapter

  print("* Check RNG reliability... ", end="")
  check_rng()
  print("OK")
  
  create_lock()
  
  import config as cfg
  cfg.pool = pool
  print("* Check framework reliability... ", end="")
  check_framework()
  print("OK")

  while True:
    cfg.pool = None
    def boundary():
      for x in ("worker_mir2", "worker_dsd2"):
        __import__(x).clean_env()
    boundary()
    del boundary
    try:
      with requests.Session() as net_sess:
        retries = Retry(total=8, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        net_sess.mount("http://", HTTPAdapter(max_retries=retries))
        net_sess.mount("https://", HTTPAdapter(max_retries=retries))
        cfg.net_sess = net_sess
        print("* Register")
        url_register = urljoin(cfg.srv_url, "/register")

        ret = cfg.net_sess.post(url_register, json={"hostname": socket.gethostname(), "version": cfg.cli_version}, timeout=8.0)
        if ret.status_code != 200:
          raise RuntimeError("Cannot connect to server: HTTP %d" % (ret.status_code,))
        ret = ret.json()
        cfg.cli_id = ret["cli_id"]
        print("* Client ID=#%d" % (cfg.cli_id,))

        with Boundary():
          print("* Enter boundary")
          worker_type, worker_config, gene_type = ret["worker_type"], ret["worker_config"], ret["gene_type"]
          print("* worker_type=%r" % (worker_type,))
          print("* worker_config=%r" % (worker_config,))
          cfg.worker_type = worker_type
          cfg.worker_config = worker_config
          cfg.gene_type = gene_type
          import geneop
          geneop.load_type(gene_type)
          
          cfg.pool = pool
          cfg.dog = watchdog.Watchdog()
          cfg.dog.start()

          worker = __import__("worker_%s" % (worker_type,))
          worker.pre_fn()
          
          while True:
            try:
              print("* Acquire GPU...")
              cfg.lock.acquire()
              print("* Got GPU!")
              print("* Asking for new task...")
              ret = send_rpc("get_task")
              action = ret["action"]
              if action== "wait":
                print("* No task available at this moment.")
                time.sleep(3 if cfg.worker_type != "toy" else 0.1)
              elif action == "run":
                print("* Got new task, working...")
                task_id, gene = ret["task_id"], ret["gene"]
                cfg.hurdle_limit_list = ret.get("hurdle_limit_list", None)
                result = worker.eval_fn(gene)
                print("* Task finished, submitting result...")
                send_rpc("submit_result", task_id, result)
                print("* Result submitted")
            finally:
              cfg.lock.release()
    except KeyboardInterrupt:
      print("* Got KeyboardInterrupt, killing watchdog...")
      if cfg.dog is not None:
        cfg.dog.request_stop = True
        cfg.dog.join()
      print("* Bye!")
      os._exit(0)
    except:
      print("* Got Exception, restarting...")
      traceback.print_exc()
      if cfg.dog is not None:
        cfg.dog.request_stop = True
        cfg.dog.join()
      print("* Restarted!")

if __name__ == "__main__":
  with mp.Pool(processes=6) as pool:
    main(pool)
