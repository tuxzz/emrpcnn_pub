class RaiseError:
  def __init__(self):
    pass
  
  def __call__(self, k):
    raise ValueError("! Missing argument `%s`" % (k,))

def get(key, default=RaiseError(), ok=str):
  import sys
  prefix = "-%s=" % (key,)
  prefix_size = len(prefix)
  for i, x in enumerate(sys.argv[1:]):
    if x.startswith(prefix):
      for j, y in enumerate(sys.argv[1:]):
        if y.startswith(prefix) and i != j:
          raise ValueError("! Conflict argument `%s`" % (y,))
      return ok(x[prefix_size:])
  if callable(default):
    return default(key)
  else:
    return default

def get_multi(key, default=RaiseError(), ok=str):
  import sys
  prefix = "-%s=" % (key,)
  prefix_size = len(prefix)
  l = [x[prefix_size:] for i, x in enumerate(sys.argv[1:]) if x.startswith(prefix)]
  if l:
    return [ok(x) for x in l]
  else:
    if callable(default):
      return default(key)
    else:
      return default

def get_switch(key):
  import sys
  s = "-%s" % (key,)
  for i, x in enumerate(sys.argv[1:]):
    if x == s:
      for j, y in enumerate(sys.argv[1:]):
        if y == s and i != j:
          raise ValueError("! Conflict argument `%s`" % (y,))
      return True
  return False