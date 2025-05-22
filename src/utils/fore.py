import os
import re
import pandas as pd
from evaluate import *

# def get_dataset_key(file_path):
#     """Infer dataset key from file path."""
#     if "webqsp" in file_path.lower():
#         return "webqsp"
#     elif "scene_graphs" in file_path.lower():
#         return "scene_graphs"
#     elif "expla" in file_path.lower():
#         return "expla_graphs"
#     else:
#         raise ValueError(f"Unknown dataset in file path: {file_path}")

# def get_eval_func(dataset_key):
#     """Map dataset key to evaluation function."""
#     if dataset_key == "webqsp":
#         return lambda pred, label: eval_hit(" ".join(pred.split("\n")), label.split("|")) == 1
#     elif dataset_key == "scene_graphs":
#         return lambda pred, label: label in pred
#     elif dataset_key == "expla_graphs":
#         return lambda pred, label: (
#             re.findall(r"support|Support|Counter|counter", pred.strip()) and
#             re.findall(r"support|Support|Counter|counter", pred.strip())[0].lower() == label
#         )
#     else:
#         raise ValueError(f"Unknown dataset key: {dataset_key}")

# def get_solvable_set_from_files(seed_paths):
#     """
#     seed_paths: list of 4 file paths (for 4 random seeds)
#     Returns: (solvable_set, total_examples)
#     """
#     assert len(seed_paths) == 4, "Must provide 4 seed file paths."

#     dfs = [pd.read_csv(path) for path in seed_paths]
#     print("  → rows:", len(dfs))
#     assert all(len(df) == len(dfs[0]) for df in dfs), "Seed files must have the same number of rows."

#     dataset_key = get_dataset_key(seed_paths[0])
#     eval_func = get_eval_func(dataset_key)
#     total_examples = len(dfs[0])
#     solvable_set = set()

#     for i in range(total_examples):
#         label = dfs[0].iloc[i]["label"]
#         correct_all = all(eval_func(df.iloc[i]["pred"], label) for df in dfs)
#         if correct_all:
#             solvable_set.add(i)

#     return solvable_set, total_examples

# def compute_fore(S_F, S_E, total_examples):
#     """
#      `Should Graph Neural Networks Use Features, Edges, Or Both?`
#      by Lukas Faber, Yifan Lu, Roger Wattenhofer
#     https://arxiv.org/abs/2103.06857

#     ForE = |S(F) ∩ S(E)| / |P|
#     """
#     return len(S_F & S_E) / total_examples

# if __name__ == "__main__":
#     # S_F = {1, 2, 3}
#     # S_E = {3, 5}
#     # total_examples = 5
#     # fore = compute_fore(S_F, S_E, total_examples)
#     # print(f"ForE score: {fore:.4f}")
#     # base_E = (
#     #     "/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/"
#     #     "model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_"
#     #     "max_txt_len_512_max_new_tokens_32_gnn_model_name_mlp_"
#     #     "gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv"
#     # )



#     # base_F = (
#     #     "/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/"
#     #     "model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_"
#     #     "max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_"
#     #     "num_epochs_10_seed{}.csv"
#     # )

#     base_E = (
#         "/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/webqsp/"
#         "model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_"
#         "max_txt_len_512_max_new_tokens_32_gnn_model_name_mlp_"
#         "gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv"
#     )

#     # Updated Model-F (GT-GNN)
#     base_F = (
#         "/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/webqsp/"
#         "model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_"
#         "max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_"
#         "patience_2_num_epochs_10_seed{}.csv"
#     )

#     # 2) list out all 4 seed files
#     seeds = [0, 1, 2, 3]
#     seed_paths_E = [base_E.format(s) for s in seeds]
#     seed_paths_F = [base_F.format(s) for s in seeds]

#     # 3) build solvable sets
#     S_E, total_E = get_solvable_set_from_files(seed_paths_E)
#     S_F, total_F = get_solvable_set_from_files(seed_paths_F)

#     # sanity check: same dataset, so same total examples
#     assert total_E == total_F, "Mismatch in number of examples between E and F files"
#     total_examples = total_E

#     # 4) compute ForE
#     fore = compute_fore(S_F, S_E, total_examples)
#     print(f"ForE score: {fore:.4f}")


import os
import re
import pandas as pd

# bring in your low‑level evaluators:
from evaluate import eval_f1, eval_acc, eval_hit

def get_dataset_key(file_path):
    if "webqsp" in file_path.lower():
        return "webqsp"
    elif "scene_graphs" in file_path.lower():
        return "scene_graphs"
    elif "expla" in file_path.lower():
        return "expla_graphs"
    else:
        raise ValueError(f"Unknown dataset in file path: {file_path}")

def get_correct_mask(path, dataset_key):
    """
    Returns a list of booleans, one per row, whether that seed
    answered correctly according to your evaluate.py logic.
    """
    # --- always read as JSONL ---
    df = pd.read_json(path, lines=True)

    if dataset_key == "webqsp":
        mask = []
        for pred, label in zip(df.pred, df.label):
            lines = pred.replace("|", "\n").split("\n")
            hit   = eval_hit(" ".join(lines), label.split("|"))
            mask.append(hit == 1)
        return mask

    elif dataset_key == "scene_graphs":
        return [label in pred for pred, label in zip(df.pred, df.label)]

    elif dataset_key == "expla_graphs":
        def row_ok(pred, label):
            m = re.findall(r"support|Support|Counter|counter", pred.strip())
            return bool(m) and m[0].lower() == label
        return [row_ok(pred, label) for pred, label in zip(df.pred, df.label)]

    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")


def get_solvable_set_from_files(seed_paths):
    """
    seed_paths: list of 4 CSV file paths (for 4 random seeds)
    Returns: (solvable_set, total_examples)
    """
    assert len(seed_paths) == 4, "Must provide 4 seed file paths."

    # infer dataset once
    dataset_key = get_dataset_key(seed_paths[0])

    # build 4 boolean masks
    masks = []
    for p in seed_paths:
        print("Loading:", p)
        mask = get_correct_mask(p, dataset_key)
        print("  → examples:", len(mask), "correct on this seed:", sum(mask))
        masks.append(mask)

    # sanity check all masks have same length
    lengths = [len(m) for m in masks]
    if len(set(lengths)) != 1:
        raise AssertionError(f"Seed files have different row counts: {lengths}")
    total = lengths[0]

    # find indices where *all* seeds agree on correct
    solvable = {i for i in range(total) if all(m[i] for m in masks)}
    return solvable, total

def compute_fore(S_F, S_E, total_examples):
    """ForE = |S(F) ∩ S(E)| / |P|"""
    return len(S_F & S_E) / total_examples

if __name__ == "__main__":
    # path‐templates for your two models (E and F):
    
    # MLP Expla
    # base_F = (
    #   "/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/"
    #   "model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_"
    #   "max_txt_len_512_max_new_tokens_32_gnn_model_name_mlp_"
    #   "gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv"
    # )
    # MLP Web
    # base_F= ("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/webqsp/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_mlp_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv")
    # Transformer Expla
    base_F=("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/model_name_gf_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_sgformer_patience_2_num_epochs_10_seed{}.csv")
    #  GCN-Edge Only
    # base_E = (
    #   "/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gcn_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv"
    # )
    # base_E=("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/webqsp/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gcn_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv")

    # GT-E Expla
    base_E= ("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv")
    
    # GT-E WebQSP
    #base_E=("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/webqsp/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv")


    
    # # Transformer Conv (Feature and Edges both)
    # base_F =("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_seed{}.csv")

    seeds = [0,1,2,3]
    paths_E = [base_E.format(s) for s in seeds]
    paths_F = [base_F.format(s) for s in seeds]

    # build solvable sets
    S_E, tot_E = get_solvable_set_from_files(paths_E)
    S_F, tot_F = get_solvable_set_from_files(paths_F)

    print(f"Size of solvable set E: {len(S_E)}")
    print(f"Size of solvable set F: {len(S_F)}")
    assert tot_E == tot_F, "Different dataset sizes between E and F!"
    total = tot_E

    # compute and print ForE
    fore = compute_fore(S_F, S_E, total)
    print(f"ForE score: {fore:.4f}")
