def count(ver, gene_int, n_ch):
  import tensorflow as tf
  from mir_util import infer
  import config as cfg
  import netop

  with cfg.ConfigBoundary(gene_ver=ver, gene_value=gene_int):
    batch_size = 1
    graph = tf.Graph()
    run_meta = tf.compat.v1.RunMetadata()
    with graph.as_default():
      x_mixed = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, cfg.frame_size // 2, 1), name="x_mixed")
      y_mixed = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 64, cfg.frame_size // 2, n_ch), name="y_mixed")
      y_pred = infer(x_mixed, n_ch, True)
      n_forward_flop = tf.compat.v1.profiler.profile(graph, run_meta=run_meta, cmd="op", options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()).total_float_ops
      y_output = tf.multiply(x_mixed, y_pred)
      loss_fn = tf.reduce_mean(input_tensor=tf.abs(y_mixed - y_output) , name="loss0")
      global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_fn, global_step=global_step)

      n_total_flop = tf.compat.v1.profiler.profile(graph, run_meta=run_meta, cmd="op", options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()).total_float_ops
      total_parameters = netop.count_parameter()
      return total_parameters, n_forward_flop

def main():
  import sys
  sys.path.append("../lib")
  import simpleopt
  n_ch = int(simpleopt.get("ch"))
  total_parameters, n_forward_flop = count(None, None, n_ch)
  print("Trainable parameters: {:,}".format(total_parameters,))
  print("Forward FLOPs: {:,}".format(n_forward_flop,))

if __name__ == "__main__":
  main()