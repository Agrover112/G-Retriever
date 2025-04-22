import os
import contextlib
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from torch_geometric.nn import TopKPooling, SAGPooling
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100

access_token = os.getenv("HF_ACCESS_TOKEN") 

class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.gnn_model_name = args.gnn_model_name
        self.gnn_in_dim = args.gnn_in_dim
        self.gnn_hidden_dim = args.gnn_hidden_dim

        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.pooling_type = args.pooling  #  ['mean', 'topk', 'sag', or 'virtual']
        self.pool_ratio = args.pool_ratio  #  ratio for pooling (e.g., 0.5)

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '64GiB', 1: '64GiB'},
            "device_map": "auto", #auto for 2
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=os.getenv("HF_ACCESS_TOKEN") ,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        # Set Pooling method.
        if self.pooling_type == 'mean':
            self.pooling = None  # Defaults to 1-graph token
        elif self.pooling_type == 'topk':
            self.pooling = TopKPooling(in_channels=args.gnn_hidden_dim, ratio=self.pool_ratio).to(self.device)
        elif self.pooling_type == 'sag':
            self.pooling = SAGPooling(in_channels=args.gnn_hidden_dim, ratio=self.pool_ratio).to(self.device)
        elif self.pooling_type == 'virtual':
            
            # Virtual nodes are processed *after* initial encoding (dim = gnn_hidden_dim)
            self.num_virtual_nodes = args.gnn_num_virtual_tokens
            self.virtual_node_emb = nn.Parameter(torch.Tensor(self.num_virtual_nodes, args.gnn_hidden_dim)).to(self.model.device)
            
            # Message Passing GNN intialization for virtual nodes
            if self.gnn_model_name == 'gcn':
                self.conv2_virtual = GCNConv(self.gnn_hidden_dim, self.gnn_hidden_dim).to(self.model.device)
                self.reset_virtual_parameters()
            elif self.gnn_model_name == 'gat':
                self.conv2_virtual = GATConv(self.gnn_hidden_dim, self.gnn_hidden_dim).to(self.model.device)
                self.reset_virtual_parameters()
            elif self.gnn_model_name == 'gt':
                self.conv2_virtual = TransformerConv(self.gnn_hidden_dim, self.gnn_hidden_dim).to(self.model.device)
                self.reset_virtual_parameters()              
            print(f"Initialized Virtual Node Pooling with k={self.num_virtual_nodes}")
            self.pooling = None
        elif self.pooling_type == 'sagM':
            self.pooling = SAGPooling(in_channels=args.gnn_hidden_dim, ratio=self.pool_ratio).to(self.device)
        elif self.pooling_type == 'none':
            self.pooling = None  # No pooling needed
            print("Using no pooling - passing all node embeddings directly")
        else:
            raise ValueError("Invalid pooling type. Choose one of: 'mean', 'topk', 'sag','sagM', 'virtual'.")

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def reset_virtual_parameters(self):
            import math
            # Simple initialization for the virtual node embeddings
            stdv = 1. / math.sqrt(self.virtual_node_emb.size(1))
            self.virtual_node_emb.data.uniform_(-stdv, stdv)
    def encode_graphs(self, samples):
        graphs = samples['graph']
        #print("Number of graphs in batch:", len(graphs))
        print("Graph nodes shape:", graphs.x.shape)
        node_counts = torch.bincount(graphs.batch)
        print("Number of nodes in each graph:", node_counts)
        print("Max number of nodes in a graph:", node_counts.max().item())
        print("Avg. number of nodes in a graph:", node_counts.float().mean().item())
        print("Graph edges shape:", graphs.edge_index.shape)
        graphs = graphs.to(self.model.device)
        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)
        print("n_embeds",n_embeds.shape)

        # Pooling: Perform pooling on nodes.
        if self.pooling_type == 'mean':
            g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')  # shape: [batch, hidden_dim]
            g_embeds = g_embeds.unsqueeze(1)  # shape becomes [batch, 1, hidden_dim]
        elif self.pooling_type in ['topk', 'sag']:
            pooled_x, _, _, batch, perm, score = self.pooling(n_embeds, graphs.edge_index, batch=graphs.batch)
            g_embeds = []  # list of tensors; each tensor shape: [num_tokens_i, hidden_dim]
            unique_batches = batch.unique(sorted=True)
            for b in unique_batches:
                tokens = pooled_x[batch == b]
                g_embeds.append(tokens)
            # At this point, you'll need to later pad these tokens per sample.
        elif self.pooling_type == 'virtual':
            k = self.num_virtual_nodes
            num_graphs = len(torch.unique(graphs.batch))
            N_real_batch = n_embeds.size(0)
            batch_vector = graphs.batch
            B = batch_vector.max().item() + 1

            # Repeat virtual node embeddings for the batch
            virtual_x = self.virtual_node_emb.repeat(B, 1) # [B * k, GnnHiddenDim]

            # Augment features
            augmented_x = torch.cat([n_embeds, virtual_x], dim=0) # [N_real + B*k, H]

            # Augment edge_index (Connect real nodes to their corresponding virtual nodes)
            real_node_indices = torch.arange(N_real_batch, device=self.device)
            virtual_node_base_indices = torch.arange(k, device=self.device)
            virtual_node_graph_start_indices = N_real_batch + batch_vector * k

            src_real_to_virtual = real_node_indices.repeat_interleave(k)
            tgt_real_to_virtual = virtual_node_base_indices.repeat(N_real_batch) + \
                                  virtual_node_graph_start_indices.repeat_interleave(k)
            edges_real_to_virtual = torch.stack([src_real_to_virtual, tgt_real_to_virtual], dim=0)
            edges_virtual_to_real = torch.stack([tgt_real_to_virtual, src_real_to_virtual], dim=0)

            augmented_edge_index = torch.cat([
                graphs.edge_index.long(),
                edges_real_to_virtual,
                edges_virtual_to_real
            ], dim=1)

            # Run conv2 on augmented graph
            augmented_x = self.conv2_virtual(augmented_x, augmented_edge_index).relu()

            # Select only the virtual node embeddings as the pooled output
            pooled_embeds = augmented_x[N_real_batch:] # [B * k, GnnHiddenDim]

            # Create the corresponding batch vector for these virtual nodes
            pooled_batch_vector = torch.arange(B, device=self.device).repeat_interleave(k) # [B * k]
            g_embeds = pooled_embeds.view(B, k, -1)

            # You have a fixed set of virtual tokens (shape: [num_virtual_tokens, hidden_dim]).
            # Replicate this tensor for every graph in the batch. At the moment this is just a placeholder for the actual one with Message Passing
            
            #g_embeds = self.conv2_virtual(g_embeds, graphs.edge_index.long(), graphs.edge_attr)
            #num_graphs = len(torch.unique(graphs.batch))
            #g_embeds = self.virtual_tokens.unsqueeze(0).repeat(num_graphs, 1, 1)  # shape: [num_graphs, num_virtual_tokens, hidden_dim]
        elif self.pooling_type == 'sagM':
            mean_embed = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
            pooled_x, _, _, batch, _, _ = self.pooling(n_embeds, graphs.edge_index, batch=graphs.batch)
            g_embeds = []
            unique_batches = batch.unique(sorted=True)
            for i in range(mean_embed.size(0)):
                # Start with mean token
                graph_tokens = [mean_embed[i]]
                
                # Add SAG tokens if this graph has any
                if i in unique_batches:
                    sag_tokens = pooled_x[batch == i]
                    #graph_tokens.append(sag_tokens)
                    graph_tokens = torch.cat([mean_embed[i].unsqueeze(0), sag_tokens], dim=0)
                # Stack tokens for this graph
                g_embeds.append(graph_tokens)
                #g_embeds.append(torch.cat(graph_tokens, dim=0))
        elif self.pooling_type == 'none':
            # Return a list of node embeddings for each graph
            g_embeds = []
            unique_batches = torch.unique(graphs.batch)
            for b in unique_batches:
                # Get nodes for this graph
                mask = (graphs.batch == b)
                graph_nodes = n_embeds[mask]
                
                # Optional: limit number of nodes to avoid excessive sequence lengths
                max_nodes = 50  # Adjust as needed
                if graph_nodes.size(0) > max_nodes:
                    # Take most important nodes by feature norm
                    node_importance = torch.norm(graph_nodes, dim=1)
                    _, indices = torch.topk(node_importance, max_nodes)
                    graph_nodes = graph_nodes[indices]
                    
                g_embeds.append(graph_nodes)
        else:
            raise ValueError("Pooling method not recognized.")

        # mean pooling
        #g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        print("Graph embeds (pooled) shape:", g_embeds.shape if isinstance(g_embeds, torch.Tensor) else [x.shape for x in g_embeds])
        #print("Graph emebeds before projection g_embeds",g_embeds.shape)

        return g_embeds
 
    def forward(self, samples):
        # encode description, questions, and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        print("Number of tokens in Question:", len(questions.input_ids[0]))
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        print("Number of tokens in Textualize:", len(descriptions.input_ids[0]))
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        print("Number of tokens in Label:", len(labels.input_ids[0]))

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device)
        )
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)
        ).unsqueeze(0)

        # encode graphs and then project graph tokens
        graph_embeds = self.encode_graphs(samples)  # This may have shape:
        # For mean pooling: [batch, 1, hidden_dim]
        # For topk/sag pooling: list of tensors, variable shapes per graph
        # For virtual pooling: [batch, num_virtual_tokens, hidden_dim]
        if self.pooling_type in ['mean', 'virtual']:
            # Apply projector elementwise: reshape to 2D, project, then reshape back
            b, t, _ = graph_embeds.shape
            graph_embeds = graph_embeds.view(b * t, -1)
            graph_embeds = self.projector(graph_embeds)
            graph_embeds = graph_embeds.view(b, t, -1)  # now shape [batch, t, 4096]
        elif self.pooling_type in ['topk', 'sag','sagM','none']:
            # For variable tokens, project each token individually.
            projected = [self.projector(x) for x in graph_embeds]
            # After projection, list elements have shape: [num_tokens_i, 4096]
            graph_embeds = projected

        #print("Graph embeds after projection:", graph_embeds.shape if isinstance(graph_embeds, torch.Tensor) else [x.shape for x in graph_embeds])

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        # For each sample, construct input embeddings; note that for topk/sag pooling you may need to pad tokens across graphs.
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = (descriptions.input_ids[i][:self.max_txt_len] +
                         questions.input_ids[i] +
                         eos_user_tokens.input_ids +
                         label_input_ids)
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # Determine graph tokens for this sample:
            if self.pooling_type in ['mean', 'virtual']:
                graph_tokens = graph_embeds[i]  # shape: [t, 4096] where t is 1 or num_virtual_tokens
            elif self.pooling_type in ['topk', 'sag', 'sagM','none']:
                graph_tokens = graph_embeds[i]  # already a tensor from the list for graph i
                
                # (Optional) You may need to pad graph_tokens to a fixed number 
                # across samples if required by your downstream LLM.
            else:
                raise ValueError("Invalid pooling type.")
            
            print("Final number of graph token:", graph_tokens.shape)
            # Concatenate BOS token, graph tokens, and standard text tokens
            inputs_embeds = torch.cat([bos_embeds, graph_tokens, inputs_embeds], dim=0)
            print("Final number of input token:", inputs_embeds.shape)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids_final = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids_final)

        # Pad the batch so that each sample has the same sequence length
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)
        print("Final number of input token after padding:", inputs_embeds.shape)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        with self.maybe_autocast():
            inference_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True  # IMPORTANT!
            )
        token_counts = [seq.shape[0] for seq in inference_ids]
        avg_tokens = sum(token_counts) / len(token_counts)
        print("Number of tokens generated:", len(token_counts))
        print("Average number of tokens generated:", avg_tokens)
        predictions = self.tokenizer.batch_decode(inference_ids, skip_special_tokens=True)
        
        #for i, seq in enumerate(inference_ids):
            #pass
            #print(f"Sample {i} generated {seq.shape[0]} tokens")
            #print(f"Sample {i} generated {self.tokenizer.decode(seq, skip_special_tokens=True)}")
        
        return outputs.loss

    def inference(self, samples):
        # Similar to forward, but without computing loss.
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        graph_embeds = self.encode_graphs(samples)
        if self.pooling_type in ['mean', 'virtual']:
            b, t, _ = graph_embeds.shape
            graph_embeds = graph_embeds.view(b*t, -1)
            graph_embeds = self.projector(graph_embeds)
            graph_embeds = graph_embeds.view(b, t, -1)
        elif self.pooling_type in ['topk', 'sag', 'sagM','none']:
            projected = [self.projector(x) for x in graph_embeds]
            graph_embeds = projected

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            if self.pooling_type in ['mean', 'virtual']:
                graph_tokens = graph_embeds[i]
            elif self.pooling_type in ['topk', 'sag', 'sagM', 'none']:
                graph_tokens = graph_embeds[i]
            inputs_embeds = torch.cat([bos_embeds, graph_tokens, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param



""" Old code for GraphLLM
    def forward(self, samples):

        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        print("Number of tokens in Question:", len(questions.input_ids[0]))
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        print("Number of tokens in Textualize:", len(descriptions.input_ids[0]))
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        print("Number of tokens in Label",len(labels.input_ids[0]))
        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)
        print("Graph embeds after projection g_embeds",graph_embeds.shape)
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            print("Final number of graph token:",graph_embeds[i].unsqueeze(0).shape)
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            print("Final number of input token:",inputs_embeds.shape)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)
        print("Final number of input token after padding:",inputs_embeds.shape)
        
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        with self.maybe_autocast():
            inference_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            attention_mask=attention_mask,
            # do_sample=True,  # Uncomment if you want sampling.
            use_cache=True  # IMPORTANT!
        )
        token_counts = [seq.shape[0] for seq in inference_ids]
        avg_tokens = sum(token_counts) / len(token_counts)
        print("Number of tokens generated:", len(token_counts))
        print("Average number of tokens generated:", avg_tokens)

        predictions = self.tokenizer.batch_decode(inference_ids, skip_special_tokens=True)
        for i, seq in enumerate(inference_ids):
            print(f"Sample {i} generated {seq.shape[0]} tokens")
            print(f"Sample {i} generated {self.tokenizer.decode(seq, skip_special_tokens=True)}")

        return outputs.loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
"""