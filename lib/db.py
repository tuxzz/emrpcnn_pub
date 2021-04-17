from typing import *

def index_musdb_type1(path:str) -> Dict[str, Dict[str, str]]:
  import os, re
  hq_exp = re.compile(r"^(.+)\.stem_(\S+)\.wav$")
  out = {}
  for root, _, file_list in os.walk(path):
    for filename in file_list:
      m = hq_exp.match(filename)
      if not m:
        continue
      name = m.group(1)
      if name in out:
        continue
      out[name] = {
        stem: os.path.join(root, "%s.stem_%s.wav") % (name, stem,) for stem in ("accompaniment", "vocals", "mix", "bass", "drums", "other")
      }
  return out

def index_musdb(path:str) -> Dict[str, Dict[str, str]]:
  import os, re
  out = {}
  _, d, _ = next(os.walk(path))
  for name in d:
    out[name] = {
      stem: os.path.join(os.path.abspath(path), name, "%s.wav") % (stem if stem != "mix" else "mixture",) for stem in ("accompaniment", "vocals", "mix", "bass", "drums", "other")
    }

  return out
