from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM
from src.model.gf_llm import GFLLM


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
    "gf_llm": GFLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "/gpfs/projects/ehpc250/ankit/llama2-7b-hf",
    "7b_chat": "/gpfs/projects/ehpc250/ankit/llama2-7b-chat-hf/",
    "13b": "/gpfs/projects/ehpc250/ankit/llama2-13b-hf",
    "13b_chat": "/gpfs/projects/ehpc250/ankit/llama2-13-chat-hf",
}
