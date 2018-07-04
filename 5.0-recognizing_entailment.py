#paper 
#Reasoning about Entailment with Neural Attention
#https://arxiv.org/pdf/1509.06664v1.pdf


import tensorflow as tf
import numpy as np

batch_size = 3
seq_len = 5
dim = 2
# [batch_size x seq_len x dim]  -- hidden states
Y = tf.constant(np.random.randn(batch_size, seq_len, dim), tf.float32)
# [batch_size x dim]            -- h_N
h = tf.constant(np.random.randn(batch_size, dim), tf.float32)

initializer = tf.random_uniform_initializer()
W = tf.get_variable("weights_Y", [dim, dim], initializer=initializer)
w = tf.get_variable("weights_w", [dim], initializer=initializer)

# [batch_size x seq_len x dim]  -- tanh(W^{Y}Y)
M = tf.tanh(tf.einsum("aij,jk->aik", Y, W))
# [batch_size x seq_len]        -- softmax(Y w^T)
a = tf.nn.softmax(tf.einsum("aij,j->ai", M, w))
# [batch_size x dim]            -- Ya^T
r = tf.einsum("aij,ai->aj", Y, a)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_val, r_val = sess.run([a, r])
    print("a:", a_val, "\nr:", r_val)
    
    
    
    
    #I came across this here https://stackoverflow.com/questions/42507030/implementing-attention-in-tensorflow
