import os
import re
import pandas as pd
from evaluate import *

def get_row_predicate(dataset_key):
    """
    Returns a function f(df) -> List[bool], where f(df)[i] is True
    iff row i is "correct" under the dataset_key's logic.
    """
    if dataset_key in ("scene_graphs", "scene_graphs_baseline"):
        return lambda df: [label in pred for pred, label in zip(df['pred'], df['label'])]

    if dataset_key == "expla_graphs":
        def f(df):
            out = []
            for pred, label in zip(df['pred'], df['label']):
                m = re.findall(r"support|Support|Counter|counter", pred.strip())
                out.append(bool(m) and m[0].lower() == label)
            return out
        return f

    if dataset_key in ("webqsp", "webqsp_baseline"):
        def f(df):
            out = []
            for pred, label in zip(df['pred'], df['label']):
                lines = pred.replace("|", "\n").split("\n")
                hit = eval_hit(" ".join(lines), label.split("|"))
                out.append(hit == 1)
            return out
        return f

    raise ValueError(f"Unknown dataset key {dataset_key}")


def get_correct_mask(path):
    """
    Reads a JSONL file from path, infers dataset_key,
    and returns a boolean list of length |P| indicating
    which examples are correct under one seed.
    """
    lk = path.lower()
    if "webqsp" in lk:
        key = "webqsp"
    elif "scene_graphs" in lk:
        key = "scene_graphs"
    elif "expla" in lk:
        key = "expla_graphs"
    else:
        raise ValueError(f"Cannot infer dataset from path: {path}")

    df = pd.read_json(path, lines=True)
    predicate = get_row_predicate(key)
    return predicate(df)


def get_solvable_set_from_files(seed_paths):
    """
    seed_paths: list of 4 file paths (for 4 random seeds)
    Returns: (solvable_set, total_examples)
    """
    assert len(seed_paths) == 4, "Must provide 4 seed file paths."

    masks = []
    for p in seed_paths:
        print(f"Loading: {p}")
        mask = get_correct_mask(p)
        print(f"  → examples: {len(mask)}, correct: {sum(mask)}")
        masks.append(mask)

    lengths = [len(m) for m in masks]
    if len(set(lengths)) != 1:
        raise AssertionError(f"Seed files have different row counts: {lengths}")
    total = lengths[0]

    solvable = {i for i in range(total) if all(m[i] for m in masks)}
    return solvable, total


def compute_fore(S_F, S_E, total_examples):
    """ForE = |S(F) ∩ S(E)| / |P|"""
    return len(S_F & S_E) / total_examples


if __name__ == "__main__":
    # Replace with your actual path templates
    # base_F= ("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_mlp_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv")
    # base_E=("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/expla_graphs/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gcn_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv")

    base_E= ("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/webqsp/model_name_graph_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_gnn_num_virtual_tokens_5_patience_2_num_epochs_10_seed{}.csv")
    base_F=("/leonardo_scratch/fast/EUHPC_D12_046/ankit/outputs/webqsp/model_name_gf_llm_pooling_mean_llm_model_name_7b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_sgformer_patience_2_num_epochs_10_seed{}.csv")
    seeds = [0, 1, 2, 3]

    paths_F = [base_F.format(s) for s in seeds]
    paths_E = [base_E.format(s) for s in seeds]

    S_F, total = get_solvable_set_from_files(paths_F)
    S_E, _     = get_solvable_set_from_files(paths_E)
    
    # Compute ForE
    fore = compute_fore(S_F, S_E, total)
    print(f"ForE score: {fore:.4f}")

    # Compute set differences and complements
    only_E    = S_E - S_F
    only_F    = S_F - S_E
    neither   = set(range(total)) - (S_E | S_F)
    not_E     = set(range(total)) - S_E
    not_F     = set(range(total)) - S_F

    print(f"|P|    ={total}" )
    print(f"|S(E)|           = {len(S_E)}")
    print(f"|S(F)|           = {len(S_F)}")
    print(f"|S(E) - S(F)|   = {len(only_E)}")
    print(f"|S(F) - S(E)|   = {len(only_F)}")
    print(f"|neither|        = {len(neither)}")
    print(f"|not S(E)|       = {len(not_E)}")
    print(f"|not S(F)|       = {len(not_F)}")