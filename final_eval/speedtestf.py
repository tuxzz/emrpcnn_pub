def test_backward(ver, gene, n_warmup=11, n_work=121):
  import tensorflow as tf
  import numpy as np
  from mir_util import infer
  import config as cfg
  import sys, time
  import netop
  sys.is_train = True

  batch_size = 1
  n_feature = 5644 // 2

  graph = tf.Graph()
  with graph.as_default():
    # Model
    p_input = tf.random.uniform((batch_size, 64, n_feature, 1), dtype=tf.float32, name="p_input")
    p_target = tf.random.uniform((batch_size, 64, n_feature, 2), dtype=tf.float32, name="p_target")
    v_pred = infer(p_input, 2, True, ver=ver, gene=gene)
    v_loss = tf.reduce_mean(input_tensor=tf.abs(p_target - v_pred), name="loss0")
    op_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(v_loss)
    
    n_param = netop.count_parameter()
    n_forward_flop = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()).total_float_ops
    print(" BWD :Total {:,} parameters in total".format(n_param))
    print(" BWD :Forward + backward operation needs {:,} FLOPS".format(n_forward_flop))

    with tf.compat.v1.Session(config=cfg.sess_cfg) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      for i_step in range(n_warmup + n_work):
        sess.run([v_loss, op_optim])
        if i_step == n_warmup:
          t = time.time()
    t_train = (time.time() - t) / n_work
    return t_train

def test_forward(ver, gene, n_warmup=11, n_work=121):
  import tensorflow as tf
  import numpy as np
  from mir_util import infer
  import config as cfg
  import sys, time
  import netop
  sys.path.append("../lib")
  sys.is_train = False

  batch_size = 1
  n_feature = 5644 // 2

  graph = tf.Graph()
  with graph.as_default():
    # Model
    print("Initialize network")
    with tf.device("/device:GPU:0"):
      p_input = tf.random.uniform((batch_size, 64, n_feature, 1), dtype=tf.float32, name="p_input")
      v_pred = infer(p_input, 2, False, ver=ver, gene=gene)
    
    n_param = netop.count_parameter()
    n_forward_flop = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()).total_float_ops
    print(" FWD :Total {:,} parameters in total".format(n_param))
    print(" FWD :Forward operation needs {:,} FLOPS".format(n_forward_flop))

    with tf.compat.v1.Session(config=cfg.sess_cfg) as sess:
      # Initialized, Load state
      sess.run(tf.compat.v1.global_variables_initializer())
      for step in range(n_warmup + n_work):
        sess.run(v_pred)
        if step == n_warmup:
          t = time.time()
    t_eval = (time.time() - t) / n_work
    return t_eval

def main():
  import sys
  import config as cfg
  with cfg.ConfigBoundary():
    t_train = test_backward(cfg.gene_ver, cfg.gene_value)
    t_eval = test_forward(cfg.gene_ver, cfg.gene_value)
    print("Train: %fbat/s" % (1 / t_train,))
    print("Eval: %fbat/s" % (1 / t_eval,))

if __name__ == "__main__":
  main()
