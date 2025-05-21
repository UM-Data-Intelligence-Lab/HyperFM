# HyperFM

HyperFM is a hyper-relational fact-centric multimodal fusion technique for link prediction tasks over multimodal hyper-relational KGs. It effectively captures the intricate interactions between different data modalities while accommodating the hyper-relational structure of the KG in a fact-centric manner via a customized Hypergraph Transformer. Please see the details in our paper below:
- Yuhuan Lu, Weijian Yu, Xin Jing, and Dingqi Yang. 2025. HyperFM: Fact-Centric Multimodal Fusion for Link Prediction over Hyper-Relational Knowledge Graphs. ACL 2025 (pp. xxxx-xxxx).

## How to run the code
###### Train and evaluate model (suggested parameters for WikiPeople and WD50K datasets)
```
python ./src/run.py --dataset "wikipeople" --device "2" --vocab_size 35005 --vocab_file "./data/wikipeople/vocab.txt" --ent2types_file "./data/wikipeople/entity2types_ttv.txt" --train_file "./data/wikipeople/train+valid.json" --test_file "./data/wikipeople/test.json" --ground_truth_file "./data/wikipeople/all.json" --num_workers 1 --num_relations 178 --num_types 3396 --max_seq_len 13 --max_arity 7 --hidden_dim 256 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --cl_batch_size 12288 --lr 5e-4 --cl_lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hyperedge_dropout 0.99 --epoch 300 --warmup_proportion 0.1

python ./src/run.py --dataset "wd50k" --device "3" --vocab_size 47688 --vocab_file "./data/wd50k/vocab.txt" --ent2types_file "./data/wd50k/entity2types_ttv.txt" --train_file "./data/wd50k/train+valid.json" --test_file "./data/wd50k/test.json" --ground_truth_file "./data/wd50k/all.json" --num_workers 1 --num_relations 531 --num_types 6320 --max_seq_len 19 --max_arity 10 --hidden_dim 256 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 512 --cl_batch_size 8192 --lr 5e-4 --cl_lr 1e-3 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hyperedge_dropout 0.8 --epoch 300 --warmup_proportion 0.1
```

# Python lib versions
This project should work fine with the following environments:
- Python 3.9.16 for data preprocessing, training and evaluation with:
    -  torch 1.10.0
    -  torch-scatter 2.0.9
    -  torch-sparse 0.6.13
    -  torch-cluster 1.6.0
    -  torch-geometric 2.1.0.post1
    -  numpy 1.23.3
- GPU with CUDA 11.3
