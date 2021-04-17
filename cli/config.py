from typing import *
import requests, watchdog
import multiprocessing as mp
from filelock import FileLock

# CLI
cli_version = "191007r0"
#srv_url = "https://evomaster:Xeiron@evo-master.tuxzz.org"
srv_url = "http://127.0.0.1:5000"

# DATASET
mir_wav_path = "../MIR-1K/Wavfile"
dsd_root_path = "../DSD100"
mus_root_path = "../musdb18_decoded/train"

# stub
cli_id: int = None
dog: watchdog.Watchdog = None
net_sess: requests.Session = None

worker_type: str = None
worker_config: Dict[str, object] = {}
pool:Optional[mp.Pool] = None
lock:Optional[FileLock] = None
