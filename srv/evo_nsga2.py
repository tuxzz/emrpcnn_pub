import sys
sys.path.append("../lib")
import geneop
from typing import *
import numpy as np
import srv_glob as glob

def checknd(me: List[float], thee: List[float]):
  return not any(v < w for v, w in zip(me, thee)) and not all(v == w for v, w in zip(me, thee))

def ndsort(l: List[List[float]], key=lambda x:x):
  size = len(l)
  S=[[] for _ in range(size)]
  front = [[]]
  n = [0] * size
  rank = [0] * size

  for p in range(size):
    S[p] = []
    n[p] = 0
    for q in range(size):
      if checknd(key(l[p]), key(l[q])):
        if q not in S[p]:
          S[p].append(q)
      elif checknd(key(l[q]), key(l[p])):
        n[p] = n[p] + 1
    if n[p] == 0:
      rank[p] = 0
      if p not in front[0]:
        front[0].append(p)

  i = 0
  while front[i]:
    Q = []
    for p in front[i]:
      for q in S[p]:
        n[q] = n[q] - 1
        if n[q] == 0:
          rank[q] = i + 1
          if q not in Q:
            Q.append(q)
    i = i + 1
    front.append(Q)
  del front[-1]
  return [[l[y] for y in x] for x in front]

def crowdsort(nd_group: List[List[float]], key=lambda x:x):
  if len(nd_group) <= 2:
    return nd_group
  l = np.asarray(list(zip(*[key(value) for value in nd_group])), dtype=np.float64)
  sorted_lidx = np.argsort(l, axis=1)
  _, n_item = sorted_lidx.shape
  dist_list = np.zeros(n_item, dtype=np.float64)
  for i_value, idx_list in enumerate(sorted_lidx):
    scale_fac = l[i_value, idx_list[-1]] - l[i_value, idx_list[0]]
    if scale_fac < 1e-10:
      scale_fac = 1.0
    dist_list[idx_list[0]] = 1e100
    dist_list[idx_list[-1]] = 1e100
    for i in range(1, n_item - 1):
      prev_idx, curr_idx, next_idx = idx_list[i - 1:i + 2]
      if dist_list[curr_idx] < 1e100:
        dist_list[curr_idx] += (l[i_value, next_idx] - l[i_value, prev_idx]) / scale_fac
  return [nd_group[idx] for idx in np.argsort(dist_list)[::-1]]

class EvoCore:
  n_min_result = 2
  n_max_result = None
  def __init__(self):
    self.population_genome = []
    self.history_genome = []
    self.mutate_prob = 0.02
    self.max_population = 25
    self.crossover_width = 999
    self.crossover_new_population = 10
    
    self.n_eval = 0
    self._status_delta_n_eval = self.n_eval
    self._status_buffer = None
    self.rnd_seed = 0x41526941
    self.rnd = np.random.RandomState(seed=self.rnd_seed)

    self.eval_fn: Callable[[List[List[bool]], List[List[float]]], List] = None

  def try_append_population(self, gene):
    for g in self.history_genome:
      if geneop.cmp_gene(g, gene):
        return False
    self.population_genome.append((gene, None, None))
    self.history_genome.append(gene)
    return True
  
  def append_random_population(self, true_prob=0.5):
    while True:
      gene = [bool(x) for x in self.rnd.choice(a=[False, True], size=(142,), p=[1 - true_prob, true_prob])]
      ok = self.try_append_population(gene)
      if ok:
        break
  
  def append_jitter_population(self, l: List[List[bool]], max_mutate_count: int):
    while True:
      gene = l[self.rnd.randint(0, len(l))][:]
      n_mutate = self.rnd.randint(1, max_mutate_count)
      for idx in self.rnd.permutation(len(gene))[:n_mutate]:
        gene[idx] = not gene[idx]
      ok = self.try_append_population(gene)
      if ok:
        break
  
  def gen_random_population(self, n_target_population: int, max_mutate_count: int):
    assert max_mutate_count > 0
    l = self.population_genome[:]
    while len(self.population_genome) < n_target_population:
      gene, _ = l[self.rnd.randint(len(l))]
      gene = gene[:]
      n_mutate = self.rnd.randint(1, max_mutate_count)
      for idx in self.rnd.permutation(len(gene))[:n_mutate]:
        gene[idx] = not gene[idx]
      self.try_append_population(gene)

  def crossover(self, n_new):
    n = len(self.population_genome)
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i, (_, x, _) in enumerate(self.population_genome):
      for j, (_, y, _) in enumerate(self.population_genome):
        dist_mat[i, j] = np.sum(np.abs(np.asarray(x) - np.asarray(y)))
    dist_idx = np.argsort(dist_mat, axis=1)
    crossover_width = self.crossover_width

    i = 0
    i_iter = 0
    n_max_iter = 100
    l = self.population_genome[:]
    while i < n_new and i_iter < n_max_iter:
      i_parent_a = self.rnd.choice(np.arange(n))
      i_parent_b = self.rnd.choice(dist_idx[i_parent_a, 1:crossover_width + 1])
      parent_a, parent_b = l[i_parent_a][0], l[i_parent_b][0]
      child = []
      for a, b in zip(parent_a, parent_b):
        child.append(a if bool(self.rnd.randint(0, 2)) else b)
      if self.try_append_population(child):
        i += 1
      i_iter += 1

  def apply_mutate(self, prob):
    cnt = 0
    for i, (gene, _, _) in enumerate(self.population_genome[:]):
      n = len(gene)
      keep_seq = self.rnd.uniform(0.0, 1.0, n) > prob
      g = gene[:]
      if not all(keep_seq):
        while True:
          g = [x if f else not x for x, f in zip(g, keep_seq)]
          if self.try_append_population(g):
            cnt += 1
            break
          else:
            self.rnd.shuffle(keep_seq)
  
  def sort_drop_population(self):
    import itertools
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
    self.rnd.seed(self.rnd_seed)
    self.crossover(self.crossover_new_population)
    self.apply_mutate(self.mutate_prob)
    self.eval_all_population()
    self.sort_drop_population()
    self.population_genome = self.population_genome[:self.max_population]
    self.rnd_seed += 1
  
  def dump(self, path):
    import pickle
    with open(path, "wb") as f:
      pickle.dump((self.population_genome, self.history_genome, self.n_eval, self._status_delta_n_eval, self._status_buffer, self.rnd_seed), f)
  
  def load(self, path):
    import pickle
    with open(path, "rb") as f:
      l = pickle.load(f)
      if len(l) == 6:
        self.population_genome, self.history_genome, self.n_eval, self._status_delta_n_eval, self._status_buffer, self.rnd_seed = l
      else:
        self.population_genome, self.history_genome, self.n_eval, self._status_delta_n_eval, self._status_buffer = l
