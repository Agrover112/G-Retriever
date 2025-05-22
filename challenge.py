import torch 
import torch_scatter 
import torch.nn as nn


# A Test file I made to practise tensor manipulation before integration.


all_node_features = torch.tensor([
    [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], # Graph 0
    [1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8], [1.9, 2.0, 2.1, 2.2] # Graph 1
], dtype=torch.float)
batch_assignment = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
num_graphs = batch_assignment.max().item() + 1
num_features = all_node_features.shape[1]
k = 3

list_of_selected_tokens_per_graph = []
for graph_idx in range(num_graphs):
    current_graph_nodes = all_node_features[batch_assignment == graph_idx]
    num_nodes_in_graph = current_graph_nodes.shape[0]
    if num_nodes_in_graph >= k:
        selected_tokens_for_graph = current_graph_nodes[:k, :]
    else:
        num_padding_needed = k - num_nodes_in_graph
        padding_vectors = torch.zeros(
            num_padding_needed, num_features, 
            dtype=all_node_features.dtype, device=all_node_features.device
        )
        selected_tokens_for_graph = torch.cat([current_graph_nodes, padding_vectors], dim=0)
    list_of_selected_tokens_per_graph.append(selected_tokens_for_graph)
final_fixed_k_tokens = torch.stack(list_of_selected_tokens_per_graph, dim=0)

# Linear expects (b,F) but we have (b,k,F)

print(final_fixed_k_tokens.view(-1,num_features).shape)

new_projected_dim=7
projector=nn.Linear(num_features,new_projected_dim)

projected_nodes_output=projector(final_fixed_k_tokens)
final_projected_graph_tokens=projected_nodes_output.view(num_graphs,k,new_projected_dim)
print(final_projected_graph_tokens.shape) # (graphs,nodes,features)

print(f"\n--- Challenge 5: Concatenating with Other Embeddings ---")
print(f"Input to Challenge 5 (final_projected_graph_tokens shape): {final_projected_graph_tokens.shape}")

bos_embedding = torch.randn(1, new_projected_dim) # Shape: [1, 7]
text_embeds_graph0 = torch.randn(5, new_projected_dim) # 5 tokens graph-1
text_embeds_graph1 = torch.randn(8, new_projected_dim) # 8 tokens graph-2
list_of_text_embeds = [text_embeds_graph0, text_embeds_graph1]

batch_input_sequences=[] # Tensor for each graph in each batch
for idx in range(num_graphs):
    curr_graph_k_tokens= final_projected_graph_tokens[idx]
    current_text_embeddings = list_of_text_embeds[idx]
    final_cat=torch.cat([bos_embedding,curr_graph_k_tokens,current_text_embeddings],dim=0)
    batch_input_sequences.append(final_cat)



print(f"\n`batch_input_sequences` contains {len(batch_input_sequences)} tensors.")
for i, seq in enumerate(batch_input_sequences):
    print(f"Shape of sequence {i}: {seq.shape}")


# --- Start of Challenge 6 ---
print(f"\n--- Challenge 6: Padding to Max Length ---")
print(f"Input to Challenge 6: `batch_input_sequences` containing {len(batch_input_sequences)} tensors with shapes:")
pad_token_embedding = torch.zeros(1, new_projected_dim) # Shape [1, 7]
print(f"Pad token embedding shape: {pad_token_embedding.shape}")


max_seq_len=0
for b in batch_input_sequences:
    if b.shape[0]> max_seq_len:
        max_seq_len=b.shape[0]

padded_batch_list=[]
for b in batch_input_sequences:
    diff=max_seq_len - b.shape[0]
    if diff >0:
        padding_tensor=pad_token_embedding.repeat(diff,1)
        padded_sequence = torch.cat([padding_tensor, b],dim=0)
    else:
        padded_sequence = b
    padded_batch_list.append(padded_sequence)

padded_batch_input_embeddings=torch.stack(padded_batch_list,dim=0)
print(f"\nFinal Batched Input Embeddings shape: {padded_batch_input_embeddings.shape}")