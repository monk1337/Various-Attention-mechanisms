#From suriyadeepan code library

def additive_attention(ref, query, ref_dim, qdim, 
        normalize=False, blend=False):
    # infer timesteps
    timesteps = tf.shape(ref)[1]
    
    U = tf.get_variable('U', 
            shape=[ref_dim, qdim],
            dtype=tf.float32, 
            initializer=tf.random_uniform_initializer(-0.01, 0.01))
    V = tf.get_variable('V', 
            shape=[qdim, qdim],
            dtype=tf.float32, 
            initializer=tf.random_uniform_initializer(-0.01, 0.01))
    Av = tf.get_variable('Av', 
            shape=[qdim, 1],
            dtype=tf.float32, 
            initializer=tf.random_uniform_initializer(-0.01, 0.01))
    # NOTE : reference should be in batch_major format
    ref_proj = tf.reshape(
        tf.matmul(tf.reshape(ref, [-1, ref_dim]), U), #  collapse dims to matmul
        [-1, timesteps, qdim]) # expand again
    hi = tf.expand_dims(tf.matmul(query, V),
                        axis=1) # expand time dim to add to reference
    
    # sum up ref, query
    blended = (ref_proj + hi)
    scores = tf.reshape(tf.matmul( 
            tf.reshape(blended, [-1, qdim]), # collapse dims
                Av), # matmul with attention vector
                  [-1, timesteps]) # attention weights across timesteps
    
    # normalize scores
    probs = tf.nn.softmax(scores)
    if normalize:
        return probs
    if blend: # reduce reference based on attention weights
        return tf.reduce_sum(ref * tf.expand_dims(probs, axis=-1), 
                axis=1) # reduce across time dimension
    return scores # return score
