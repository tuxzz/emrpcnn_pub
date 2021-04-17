import sys
sys.path.append("../lib")
import geneop
from typing import *
import numpy as np
import srv_glob as glob

class EvoCore:
  n_min_result = 2
  n_max_result = None
  def __init__(self):
    self.population_genome = []
    self.history_genome = []
    self.mutate_prob = 0.02
    self.max_population = 25
    
    self.n_eval = 0
    self._status_delta_n_eval = self.n_eval
    self._status_buffer = None
    self.rnd_seed = 0x41526941
    self.rnd = np.random.RandomState(seed=self.rnd_seed)
    self.ap = False

    self.eval_fn: Callable[[List[List[bool]], List[List[float]]], List] = None

  def try_append_population(self, gene):
    for g in self.history_genome:
      if geneop.cmp_gene(g, gene):
        return False
    self.population_genome.append((gene, None, None))
    self.history_genome.append(gene)
    return True
  
  def append_jitter_population(self, l: List[List[bool]], max_mutate_count: int):
    if self.ap:
      return
    assert len(l) == 1
    self.ap = True
    n = 0
    for x in range(2250):
      while not self.try_append_population(self.gen_mutate(l[0], self.mutate_prob)):
        pass
      n += 1
    glob.print_status("RANDOM", "* generated %d individuals" % (n,))
  
  def append_random_population(self, *args, **kwargs):
    return

  def gen_mutate(self, gene, prob):
    n = len(gene)
    while True:
      keep_seq = self.rnd.uniform(0.0, 1.0, n) > prob
      if all(keep_seq):
        continue
      return [x if f else not x for x, f in zip(gene[:], keep_seq)]
  
  def sort_drop_population(self):
    import itertools
    self.dump("./undropped_random.pickle")
    nd_list = ndsort(self.population_genome, key=lambda x:x[1])
    self.population_genome = list(itertools.chain(*(crowdsort(x, key=lambda x:x[1]) for x in nd_list)))
    self.population_genome = self.population_genome[:self.max_population]
    self._status_buffer = list(np.max([(x[1], x[2]) for x in self.population_genome], axis=0))
  
  def eval_all_population(self):
    assert callable(self.eval_fn), "Evaluate function must be callable"
    to_eval = [gene for gene, score, _ in self.population_genome if score is None]
    idx_list = [i for i, (_, score, _) in enumerate(self.population_genome) if score is None]
    result_list, valid_score_list = self.eval_fn(to_eval)
    for idx, score, valid_score in zip(idx_list, result_list, valid_score_list):
      self.population_genome[idx] = (self.population_genome[idx][0], score, valid_score)
      self.n_eval += 1
  
  def print_status(self):
    glob.print_status("MASTER", " :extremum_score=%s, n_eval=%d, delta_n_eval=%d" % (str(self._status_buffer), self.n_eval, self.n_eval - self._status_delta_n_eval))
    self._status_delta_n_eval = self.n_eval

  def init_step(self):
    self.rnd.seed(self.rnd_seed)
    glob.print_status("MASTER", "* Initial evaluate for all population")
    self.eval_all_population()
    self.sort_drop_population()
    self.rnd_seed += 1

  def step(self):
    assert False, "No"
  
  def dump(self, path):
    import pickle
    with open(path, "wb") as f:
      pickle.dump((self.population_genome, self.history_genome, self.n_eval, self._status_delta_n_eval, self._status_buffer, self.rnd_seed), f)
  
  def load(self, path):
    import pickle
    self.ap = True
    with open(path, "rb") as f:
      l = pickle.load(f)
      if len(l) == 6:
        self.population_genome, self.history_genome, self.n_eval, self._status_delta_n_eval, self._status_buffer, self.rnd_seed = l
      else:
        self.population_genome, self.history_genome, self.n_eval, self._status_delta_n_eval, self._status_buffer = l
