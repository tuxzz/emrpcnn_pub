import threading
from typing import Dict
import pickle, collections

def print_dict(x: Dict[str, object]):
  import srv_glob as glob
  for k, v in x.items():
    glob.print_status("MASTER", " :%s=%r" % (k, v))

def print_config():
  import srv_glob as glob
  import config as cfg
  glob.print_status("MASTER", "* Evolution config")
  x = {
    "n_gen": cfg.n_gen,
    "evo_type": cfg.evo_type,
    "gene_type": cfg.gene_type,
    "init_random_ratio": cfg.init_random_ratio,
    "p2": cfg.p2,
  }
  print_dict(x)
  glob.print_status("MASTER", "* Worker config")
  glob.print_status("MASTER", " :worker_type=%r" % (cfg.worker_type,))
  print_dict(cfg.worker_config)

def dict_to_tuple(d):
  return tuple(sorted((x for x in d.items()), key=lambda x:x[0]))

class Master(threading.Thread):
  def __init__(self):
    import sys, os
    import srv_glob as glob
    import config as cfg
    super().__init__(name="Evo Master")
    if cfg.worker_cache_path is not None:
      if os.path.exists(cfg.worker_cache_path):
        glob.print_status("MASTER", "* Loaded worker cache")
        with open(cfg.worker_cache_path, "rb") as f:
          self.worker_cache = pickle.load(f)
      else:
        glob.print_status("MASTER", "* Worker cache not found")
        self.worker_cache = collections.defaultdict(dict)
    else:
      glob.print_status("MASTER", "* Worker cache is disabled")
      self.worker_cache = collections.defaultdict(dict)
  
  def run(self):
    import config as cfg
    import srv_glob as glob
    import geneop, pickle, os
    from typing import List
    from datetime import datetime

    glob.print_status("MASTER", "* Evolution Master started at %s" % (datetime.utcnow().strftime("%m/%d/%Y %H:%M:%S UTC"),))
    print_config()
    
    def eval_fn(gene_list: List[List[bool]]):
      # Query if gene is cached
      container_key = (cfg.worker_type, dict_to_tuple(cfg.worker_config))
      result_container = self.worker_cache[container_key]
      query_result = [result_container.get(geneop.cvtlstint(gene), None) for gene in gene_list]
      for result in query_result:
        if result is not None:
          glob.print_status("EVAL", "score=%s (cached)" % (str(result),))
      missed_gene_list = [gene for gene, result in zip(gene_list, query_result) if result is None]
      missed_gene_idx_list = [i for i, result in enumerate(query_result) if result is None]
      if missed_gene_list:
        # Send task to workers
        glob.task_input_queue.put(missed_gene_list)
        worker_result_list = glob.task_output_queue.get()
        glob.incoming_population.clear()
        # Combine result
        for idx, gene, result in zip(missed_gene_idx_list, missed_gene_list, worker_result_list):
          query_result[idx] = result
          result_container[geneop.cvtlstint(gene)] = result
        # Write cache
        if cfg.worker_cache_path is not None:
          if cfg.worker_cache_backup_path is not None and os.path.exists(cfg.worker_cache_path):
            # backup old cache, assume filesystem metadata operation is atomic
            if os.path.exists(cfg.worker_cache_backup_path):
              os.remove(cfg.worker_cache_backup_path)
            os.rename(cfg.worker_cache_path, cfg.worker_cache_backup_path)
          with open(cfg.worker_cache_path, "wb") as f:
            pickle.dump(self.worker_cache, f)
      out = tuple(tuple(result[idx] for idx in cfg.result_needed) for result in query_result)
      out_valid = tuple(result[cfg.valid_idx] for result in query_result)
      return out, out_valid
  
    evo = cfg.evo_module.EvoCore()
    evo.mutate_prob = cfg.p2
    glob.evo_obj = evo
    evo.eval_fn = eval_fn
    i_start_gen = 0
    if os.path.exists(cfg.status_path):
      glob.print_status("MASTER", " :Load evolution status")
      with open(cfg.status_path, "rb") as f:
        bkpt_gen = pickle.load(f)
      bkpt_path = cfg.init_status_path if bkpt_gen == -1 else cfg.gen_status_path % (bkpt_gen,)
      i_start_gen = bkpt_gen + 1
      evo.load(bkpt_path)
      glob.create_plot()
    else:
      n_init_population = evo.max_population * 3 // 2
      n_random_population = int(cfg.init_random_ratio * n_init_population)
      n_manual_population = n_init_population - n_random_population

      if n_manual_population > 0:
        for x in cfg.manual_gene_list:
          if not evo.try_append_population(x):
            glob.print_status("MASTER", "ERROR: Cannot append an initial population.")
          del x
        for _ in range(1, n_manual_population):
          evo.append_jitter_population(cfg.manual_gene_list, cfg.gene_jitter_count)
      for i in range(n_random_population):
        evo.append_random_population()
      glob.print_status("MASTER", "* Got %d initial population (%d manual jitter + %d random)" % (n_init_population, n_manual_population, n_random_population))
      evo.init_step()
      glob.create_plot()
      evo.dump(cfg.init_status_path)
      with open(cfg.status_path, "wb") as f:
        pickle.dump(-1, f)
      evo.print_status()
    for i_gen in range(i_start_gen, cfg.n_gen):
      glob.print_status("MASTER", "* [Gen %d/%d] at %s" % (i_gen, cfg.n_gen, datetime.utcnow().strftime("%m/%d/%Y %H:%M:%S UTC"),))
      if cfg.evo_type == "nsga2" or cfg.evo_type == "random":
        glob.prev_population = [(v, s[1]) for g, s, v in evo.population_genome]
      else:
        glob.prev_population = [(v,) for g, s, v in evo.population_genome]
      evo.step()
      glob.create_plot()
      evo.dump(cfg.gen_status_path % (i_gen,))
      with open(cfg.status_path, "wb") as f:
        pickle.dump(i_gen, f)
      evo.print_status()
    glob.print_status("MASTER", "* Everything done")
    os._exit(0)
