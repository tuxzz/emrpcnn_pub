from typing import List, Dict, Tuple, Callable

def load_type(v: str):
  import geneop
  try:
    geneop.build = __import__("build_%s" % (v,))
    geneop.build_from_gene = geneop.build.build_from_gene
    geneop.gene_len = geneop.build.gene_len
  except ModuleNotFoundError:
    print("* No module `build`")
  try:
    geneop.cmp = __import__("cmp_%s" % (v,))
    geneop.cmp_gene = geneop.cmp.cmp_gene
  except ModuleNotFoundError:
    print("* No module `cmp`")

def cvtintlst(x: int, n: int) -> List[bool]:
  x = int(x)
  n = int(n)
  return [((1 << i) & x) != 0 for i in range(n)[::-1]]

def cvtlstint(l: List[bool]) -> int:
  l = [bool(x) for x in l]
  x = 0
  for _, v in enumerate(l):
    x <<= 1
    x |= v
  return x

def cvtlstgray(l: List[bool]) -> int:
  l = [bool(x) for x in l]
  x = 0
  for i, v in enumerate(l):
    x <<= 1
    if i != 0:
      v = v ^ p
    x |= v
    p = v
  return x