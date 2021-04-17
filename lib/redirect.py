class ConsoleAndFile:
  __slots__ = ("path", "file",)
  def __init__(self, path):
    self.path = path
    self.file = None

  def __enter__(self):
    self.file = open(self.path, "wb")
    return self

  def __exit__(self, tp, value, trace):
    self.file.close()
    self.file = None
  
  def print(self, *args, **kwargs):
    assert self.file is not None, "File is not opened"
    assert "file" not in kwargs, "Keyword argument `file` is not allowed"
    import io
    f = io.StringIO()
    print(*args, **kwargs, file=f)
    s = f.getvalue()
    print(s, end="")
    self.file.write(s.encode("utf-8"))

class ConsoleToBuffer:
  def __init__(self, redirect_stdout=True, redirect_stderr=True):
    import sys, io
    self.real_stdout = sys.stdout
    self.real_stderr = sys.stderr
    self.fake_stdout = io.StringIO() if redirect_stdout else self.real_stdout
    self.fake_stderr = io.StringIO() if redirect_stderr else self.real_stderr
    
  def __enter__(self):
    import sys
    sys.stdout = self.fake_stdout
    sys.stderr = self.fake_stderr
    return self

  def __exit__(self, tp, value, trace):
    import sys
    sys.stdout = self.real_stdout
    sys.stderr = self.real_stderr