#after getting output from bidirectional rnn 

#Attention_layer 
        
        x_attention = tf.reshape(transpose,[-1,rnn_num_units*2])
        attention_size=tf.get_variable(name='attention',shape=[rnn_num_units*2,1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        bias_ = tf.get_variable(name='bias_',shape=[1],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        linear_projection = tf.add(tf.matmul(x_attention,attention_size),bias_)
#         print(sentence_input.shape[0])
        reshape_ = tf.reshape(linear_projection,[tf.shape(sentence_input)[0],tf.shape(sentence_input)[1],-1])
        attention_output=tf.nn.softmax(reshape_,dim=1)
        
        atten_visualize=tf.reshape(attention_output,[tf.shape(sentence_input)[0],tf.shape(sentence_input)[1]],name='plot_dis')
        
        multi = tf.multiply(attention_output,transpose)
        

        atten_out_s = tf.reduce_sum(multi,1)

#         attention_visualize = tf.reshape(atten_out,[tf.shape(sentence_input)[0],tf.shape(sentence_input)[1]])
