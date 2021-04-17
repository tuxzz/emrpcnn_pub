from typing import *

class EvalScriptLine:
  __slots__ = ("i_line", "raw_text", "gene_type", "evo_type", "eval_type", "gene_value", "alias_name_list")
  def __init__(self, i_line:int, raw_text:str, gene_type:str, evo_type:str, eval_type:str, gene_value:int, alias_name_list:Iterable[str]):
    self.i_line = i_line
    self.raw_text = raw_text
    self.gene_type = gene_type
    self.evo_type = evo_type
    self.eval_type = eval_type
    self.gene_value = gene_value
    self.alias_name_list = alias_name_list

def load_eval_script(path):
  l = []
  alias_set = set()
  with open(path, "r") as f:
    for i_line, line in enumerate(f.readlines()):
      line = line.strip()
      if (not line) or (line[0] == "!"):
        continue
      lsp = [x.strip() for x in line.split(" ") if x.strip()]
      gene_type, evo_type, eval_type, gene_value = lsp[:4]
      alias_name_list = lsp[4:]
      if gene_type in ("sa",):
        print("* Skipped `sa`")
        continue
      assert gene_type in ("v1", "sa2", "shn", "mrfcnn",)
      assert evo_type in ("g1", "nsga2", "any",)
      assert eval_type in ("mir2", "dsd2", "dsd4",)
      for x in alias_name_list:
        assert (eval_type, x) not in alias_set
        alias_set.add((eval_type, x))
      x = EvalScriptLine(i_line, line, gene_type, evo_type, eval_type, int(gene_value), alias_name_list)
      l.append(x)
    return l