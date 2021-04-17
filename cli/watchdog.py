import threading

class Watchdog(threading.Thread):
  def __init__(self):
    super().__init__(name="Watchdog")
    self.heart_rate = 30.0
    self.request_stop = False
  
  def start(self):
    import time
    self.request_stop = False
    super().start()
  
  def run(self):
    import config, requests, time, sys
    from requests.compat import urljoin
    url_rpc = urljoin(config.srv_url, "/rpc/%d" % (config.cli_id,))
    t = -9999999999.0
    while not self.request_stop:
      now = time.time()
      if now - t >= self.heart_rate:
        ret = config.net_sess.put(url_rpc, json={"cmd": "heartbeat"})
        if ret.status_code != 200:
          print("Heartbeat failed: HTTP %d(%s)" % (ret.status_code, ret.text), file=sys.stderr)
          break
        t = time.time()
      time.sleep(0.1)
