import matplotlib as mpl
mpl.use("svg")
import pylab

from flask import Flask, Response, request, send_file
import srv_glob as glob
import json
import config as cfg

def check_rng():
  import random
  import numpy as np
  random.seed(0x41526941)
  np.random.seed(0x41526941)
  l = [1, 2, 3, 4, 5]
  random.shuffle(l)
  assert l == [2, 3, 5, 1, 4], "invalid rng"
  assert (np.random.permutation(5) == [0, 1, 3, 4, 2]).all(), "invalid rng"
print("* Check RNG reliability... ", end="")
check_rng()
print("OK")

app = Flask(__name__)
index_data = open("console.html", "rb").read().decode("utf-8")

@app.route("/update", methods=["GET"])
def report_status():
  import time
  t = time.time()
  with glob.glob_lock:
    glob.clear_zomibes(t)
  obj = {
    "console_data": glob.status_strbuffer,
    "cli_data": [(k, t - v["last_seen"], v["hostname"], v["curr_task"] is not None) for k, v in glob.online_client_dict.items()]
  }
  return Response(json.dumps(obj), mimetype="application/json", status=200)

@app.route("/", methods=["GET"])
def index():
  return Response(index_data, mimetype="text/html", status=200)

@app.route("/chart.svg", methods=["GET"])
def chart():
  import os
  if os.path.exists("./chart.svg"):
    return send_file("./chart.svg")
  else:
    return Response("No chart available at this moment", mimetype="text/plain", status=404)

@app.route("/register", methods=["POST"])
def register_request():
  import random, time
  import config as cfg
  hostname = request.json.get("hostname", None)
  if hostname is None:
    return Response("Invalid hostname", mimetype="text/plain", status=400)
  version = request.json.get("version", None)
  if version not in cfg.allowed_cli_version:
    return Response("Invalid veriosn", mimetype="text/plain", status=400)
  while True:
    cli_id = random.randint(0, 4294967295)
    if not cli_id in glob.online_client_dict:
      glob.online_client_dict[cli_id] = {
        "last_seen": time.time(),
        "curr_task": None,
        "hostname": hostname,
      }
      break
  print("Register %d" % cli_id)
  out = {
    "cli_id": cli_id,
    "worker_type": cfg.worker_type,
    "worker_config": cfg.worker_config,
    "gene_type": cfg.gene_type,
  }
  return Response(json.dumps(out), mimetype="application/json", status=200)

@app.route("/rpc/<int:cli_id>", methods=["PUT"])
def rpc_post(cli_id: int):
  import time, json, os
  print("RPC from %d" % cli_id)
  with glob.glob_lock:
    t = time.time()
    glob.clear_zomibes(t)
    cli = glob.online_client_dict.get(cli_id, None)
    req_obj = request.get_json()
    cmd = req_obj.get("cmd", None)
    args = req_obj.get("args", [])
    if cli is None and cmd != "bye":
      return Response("Client doesn't exist", mimetype="text/plain", status=400)
    elif cli is None:
      glob.print_status("SRV", "Unexpected bye-bye from Client %d!" % (cli_id,))
      return Response("Success", mimetype="text/plain", status=200)
    if cmd is None:
      return Response("Bad request", mimetype="text/plain", status=400)
    cli["last_seen"] = t
    if cmd == "heartbeat":
      print("Heartbeat")
      pass
    elif cmd == "submit_result":
      print("Submit")
      if len(args) != 2 and not (isinstance(args[0], int) and isinstance(args[1], (list, tuple))):
        return Response("Invalid argument", mimetype="text/plain", status=400)
      if cli["curr_task"] is None:
        glob.print_status("SRV", "Client %d doesn't need submit anything!" % (cli_id,))
        #return Response("Nothing to submit", mimetype="text/plain", status=400)
      else:
        glob.set_result(args[0], args[1])
        cli["curr_task"] = None
    elif cmd == "get_task":
      print("Get task")
      if cli["curr_task"] is not None:
        glob.print_status("SRV", "Client %d re-requested a task!" % (cli_id,))
        task_obj = cli["curr_task"]
      else:
        status, obj = glob.get_task(cli_id)
        if status == "wait":
          return Response(json.dumps({
            "action": "wait",
          }), mimetype="application/json", status=200)
        else:
          task_id, gene = obj
          task_obj = {
            "action": "run",
            "task_id": task_id,
            "gene": gene,
          }
          cli["curr_task"] = task_obj
      return Response(json.dumps(task_obj), mimetype="application/json", status=200)
    elif cmd == "bye":
      print("Bye")
      if cli["curr_task"] is not None:
        glob.unassign_task(cli["curr_task"]["task_id"])
      del glob.online_client_dict[cli_id]
    else:
      return Response("Unknown command", mimetype="text/plain", status=400)
    return Response("Success", mimetype="text/plain", status=200)

def main():
  import master, os
  try:
    if os.path.exists("./chart.svg"):
      os.remove("./chart.svg")
    glob.master_obj = master.Master()
    glob.master_obj.start()
    app.run(host=cfg.listen_addr, port=cfg.listen_port)
  except:
    import traceback
    traceback.print_exc()
    os._exit(0)

if __name__ == '__main__':
  main()
