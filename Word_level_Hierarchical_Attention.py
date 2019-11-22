def attention_word_level(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # 0) one layer of feed forward network
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,
                                                          self.hidden_size * 2])  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2];W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_word) + self.W_b_attention_word)  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,
                                                                         self.hidden_size * 2])  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # attention process:1.get logits for each word in the sentence. 2.get possibility distribution for each word in the sentence. 3.get weighted sum for the sentence as sentence representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vecotor_word)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[batch_size*num_sentences,sequence_length]
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # shape:[batch_size*num_sentences,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max)  # shape:[batch_size*num_sentences,sequence_length]
        # 3) get weighted hidden state by attention vector
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[batch_size*num_sentences,sequence_length,1]
        # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation,
                                                axis=1)  # shape:[batch_size*num_sentences,hidden_size*2]
        return sentence_representation  # shape:[batch_size*num_sentences,hidden_size*2]
