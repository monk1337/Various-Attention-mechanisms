def attention_sentence_level(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[None,hidden_size*4]
        input2: sentence level context vector:[self.hidden_size*2]
        :return:representation.shape:[None,hidden_size*4]
        """
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)  # shape:[None,num_sentence,hidden_size*4]

        # 0) one layer of feed forward
        hidden_state_2 = tf.reshape(hidden_state_,
                                    shape=[-1, self.hidden_size * 4])  # [None*num_sentence,hidden_size*4]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_sentence) + self.W_b_attention_sentence)  # shape:[None*num_sentence,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,
                                                                         self.hidden_size * 2])  # [None,num_sentence,hidden_size*2]
        # attention process:1.get logits for each sentence in the doc.2.get possibility distribution for each sentence in the doc.3.get weighted sum for the sentences as doc representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_sentence)  # shape:[None,num_sentence,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[None,num_sentence]. that is get logit for each num_sentence.
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)  # shape:[None,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)  # shape:[None,num_sentence]
        # 3) get weighted hidden state by attention vector(sentence level)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[None,num_sentence,1]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[None,num_sentence,hidden_size*2]<---p_attention_expanded:[None,num_sentence,1];hidden_state_:[None,num_sentence,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)  # shape:[None,hidden_size*2]
        return sentence_representation  # shape:[None,hidden_size*2]
