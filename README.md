# Improving Temporal Graph Networks for Dynamic Link Prediction 

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing

#### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.

#### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```

### Model Training

Link-prediction experiments done by running following commands:
```{bash}
### Wikipedia ###
# TGN-agg_last: 
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --aggregator last

# TGN-agg_mean: 
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-agg_attn: 
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --aggregator attn


### Reddit ###
# TGN-agg_last: 
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --aggregator last -d reddit

# TGN-agg_mean-reddit: 
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10 -d reddit

# TGN-agg_attn-reddit: 
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --aggregator attn -d reddit

```

#### General flags

```{txt}
optional arguments:
  -d DATA, --data DATA         Data sources to use (wikipedia or reddit)
  --bs BS                      Batch size
  --prefix PREFIX              Prefix to name checkpoints and results
  --n_degree N_DEGREE          Number of neighbors to sample at each layer
  --n_head N_HEAD              Number of heads used in the attention layer
  --n_epoch N_EPOCH            Number of epochs
  --n_layer N_LAYER            Number of graph attention layers
  --lr LR                      Learning rate
  --patience                   Patience of the early stopping strategy
  --n_runs                     Number of runs (compute mean and std of results)
  --drop_out DROP_OUT          Dropout probability
  --gpu GPU                    Idx for the gpu to use
  --node_dim NODE_DIM          Dimensions of the node embedding
  --time_dim TIME_DIM          Dimensions of the time embedding
  --use_memory                 Whether to use a memory for the nodes
  --embedding_module           Type of the embedding module
  --message_function           Type of the message function
  --memory_updater             Type of the memory updater
  --aggregator                 Type of the message aggregator
  --memory_update_at_the_end   Whether to update the memory at the end or at the start of the batch
  --message_dim                Dimension of the messages
  --memory_dim                 Dimension of the memory
  --backprop_every             Number of batches to process before performing backpropagation
  --different_new_nodes        Whether to use different unseen nodes for validation and testing
  --uniform                    Whether to sample the temporal neighbors uniformly (or instead take the most recent ones)
  --randomize_features         Whether to randomize node features
  --dyrep                      Whether to run the model as DyRep
```