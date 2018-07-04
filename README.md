# Various-Attention-mechanisms
This repository contain various types of attention mechanism like Bahdanau , Soft attention , Additive Attention , Hierarchical Attention etc


Luong attention and Bahdanau attention


Luong attention and Bahdanau attention are two popluar attention mechanisms. These two attention mechanisms are similar except:

In Luong attention alignment at time step t is computed by using hidden state at time step t, h⃗ t and all source hidden states, whereas in Bahdanau attention hidden state at time step t-1, h⃗ t−1 is used.

To integrate context vector c⃗ t, Bahdanau attention chooses to concatenate it with hidden state h⃗ t−1 as the new hidden state which is fed to next step to generate h⃗ t as well as predict yt+1. Luong attention instead creates an independent RNN-like structure to take the concatenatation of c⃗ t and h⃗ t as input and output h̃ ⃗ t which serves to predict yt and add additional input features.

