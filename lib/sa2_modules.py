# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''

import numpy as np
import tensorflow as tf

def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(x=inputs, axes=[-1], keepdims=True)
        beta= tf.compat.v1.get_variable("beta", params_shape, initializer=tf.compat.v1.zeros_initializer())
        gamma = tf.compat.v1.get_variable("gamma", params_shape, initializer=tf.compat.v1.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.compat.v1.variable_scope("shared_weight_matrix"):
        embeddings = tf.compat.v1.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

def scaled_dot_product_attention(scope,Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(a=K, perm=[0,1, 3, 2]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, Q, K, type="key")

        # causality or future blinding masking
        #if causality:
        #    outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(a=outputs, perm=[0,1, 3, 2])
        #tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking
        outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def mask(inputs, queries=None, keys=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)

    e.g.,
    >> queries = tf.constant([[[1.],
                        [2.],
                        [0.]]], tf.float32) # (1, 3, 1)
    >> keys = tf.constant([[[4.],
                     [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs = tf.constant([[[4., 0.],
                               [8., 0.],
                               [0., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "key")
    array([[[ 4.0000000e+00, -4.2949673e+09],
        [ 8.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
    >> inputs = tf.constant([[[1., 0.],
                             [1., 0.],
                              [1., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "query")
    array([[[1., 0.],
        [1., 0.],
        [0., 0.]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(keys), axis=-1))  # (N, T_k)
        masks = tf.expand_dims(masks, 2) # (N, 1, T_k)
        masks = tf.tile(masks, [1,1, tf.shape(input=queries)[2], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.compat.v1.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(input_tensor=tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1,1, 1, tf.shape(input=keys)[2]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs*masks
    elif type in ("f", "future", "right"):
        print("error future masking!")

    else:
        print("Check if you entered type correctly!")


    return outputs

def multihead_attention(queries, keys, values,scope,
                        num_heads=1, 
                        dropout_rate=0,
                        training=False,
                        causality=False):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Linear projections
        Q = tf.compat.v1.layers.dense(queries, d_model, use_bias=False) # (N, T_q, d_model)
        K = tf.compat.v1.layers.dense(keys, d_model, use_bias=False) # (N, T_k, d_model)
        V = tf.compat.v1.layers.dense(values, d_model, use_bias=False) # (N, T_k, d_model)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=3), axis=1) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=3), axis=1) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=3), axis=1) # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(scope,Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=1), axis=3 ) # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
 
    return outputs


def multihead_attention_tran(queries, keys, values,name):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''

    outputs=multihead_attention(tf.transpose(a=queries, perm=[0,2, 1, 3]), tf.transpose(a=keys, perm=[0,2, 1, 3]), tf.transpose(a=values, perm=[0,2, 1, 3]), name,1)


    return tf.transpose(a=outputs, perm=[0,2, 1, 3])





def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Inner layer
        outputs = tf.compat.v1.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.compat.v1.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)
    
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(input=inputs)[0], tf.shape(input=inputs)[1] # dynamic
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(value=position_enc, dtype=tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(params=position_enc, ids=position_ind)

        # masks
        if masking:
            outputs = tf.compat.v1.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.cast(outputs, dtype=tf.float32)

def att(en, dec, name):
    out = multihead_attention_tran(en, dec, dec, name)
    return out
