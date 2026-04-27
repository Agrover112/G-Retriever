# GRetPool

<img width="3260" height="1568" alt="image" src="https://github.com/user-attachments/assets/8479a1d9-f0b1-4186-b63c-480a25655007" />

## Environment setup
```
conda create --name g_retriever python=3.9 -y
conda activate g_retriever

# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install peft
pip install pandas
pip install ogb
pip install transformers
pip install wandb
pip install sentencepiece
pip install torch_geometric
pip install datasets
pip install pcst_fast
pip install gensim
pip install scipy==1.12
pip install protobuf
```

    # Required modules to be loaded
	    module load intel impi hdf5 mkl cuda/11.8 cudnn/9.0.0-cuda11 anaconda/2024.02 
	# Activate Conda environment
		micromamba activate micromamba/envs/g_ret/
## Download the Llama 2 Model
1. Go to Hugging Face: https://huggingface.co/meta-llama/Llama-2-7b-hf. You will need to share your contact information with Meta to access this model.
2. Sign up for a Hugging Face account (if you don’t already have one).
3. Generate an access token: https://huggingface.co/docs/hub/en/security-tokens.
4. Add your token to the code file as follows:
  ```
  From transformers import AutoModel
  access_token = "hf_..."
  model = AutoModel.from_pretrained("private/model", token=access_token)
  ```




## Data Preprocessing
```
# expla_graphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

# scene_graphs, might take
python -m src.dataset.preprocess.scene_graphs
python -m src.dataset.scene_graphs

# webqsp
python -m src.dataset.preprocess.webqsp
python -m src.dataset.webqsp
```

## Experiments

This command showcases basic arguments that can be used for replicating experiments.
    
    python train.py \
    --dataset <dataset> --model_name <model_type> --llm_frozen <True/False> \
    --lora_r  <rank>  --lora_alpha  <2*rank> --lora_dropout <dropout> \
    --pooling <graph_pooling_operator> \
    --pool_ratio <compression>
    --gnn_num_virtual_tokens  <GNN_ReadOut_Tokens> \
    --seed  $seed

## Pooling 

The most basic approach is to either use 1 or all nodes as tokens. This is implemented using the `mean` pooling and using `all` options.

    python train.py \
    --dataset expla_graphs --model_name graph_llm --llm_frozen False \
    --lora_r  4  --lora_alpha  8  --lora_dropout  0.05 \
    --pooling mean/all

There are two different types of graph pooling methods. One which perform pooling via clustering qand the other that perform pooling via pruning of nodes. Each of them however have different ways of combining node features after respective clustering or pruning step.
In each case the arguments passed differ.

###  Aggregation
For **Clustering**, available `--pooling` options are `diffpool,mincutpool,randk,virtual`
and we specify number of clusters using 
`-- gnn_num_virtual_tokens`.

    python train.py \
    --dataset expla_graphs --model_name graph_llm --llm_frozen False \
    --lora_r  4  --lora_alpha  8  --lora_dropout  0.05 \
    --pooling diffpool --gnn_num_virtual_tokens  8 

Note: Randk is simply selecting K random nodes, so it follows similar arguments but isn't performing any clustering. 


### Pruning
For **Pruning**, available `-- pooling` options are: `topk,sag`and we specify the pruning ratio  `-- pool_ratio` based on avg. no of nodes per graph for a given dataset

    python train.py \
    --dataset expla_graphs --model_name graph_llm --llm_frozen False \
    --lora_r  4  --lora_alpha  8  --lora_dropout  0.05 \
    --pooling sag --pool_ratio  1 

In both cases we can keep an approximate number of output tokens by manipulating these arguments. A `--pool_ratio` 1 is used for ExplaGraphs and 0.44 is used for WebQSP.

## Citation
```

```



