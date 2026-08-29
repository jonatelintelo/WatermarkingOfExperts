from dataclasses import dataclass


@dataclass
class MoEModelConfig:
    model_name: str
    num_experts: int
    num_moe_layers: int
    top_k: int
    gate_name: str
    expert_templates: list[str]


MODELS_CONFIG = {
    "allenai/OLMoE-1B-7B-0125-Instruct": MoEModelConfig(
        model_name="OLMoE-1B-7B-0125-Instruct",
        num_experts=64,
        num_moe_layers=16,
        top_k=8,
        gate_name="mlp.gate",
        expert_templates=[
            "model.layers.{layer}.mlp.experts.{expert}.gate_proj",
            "model.layers.{layer}.mlp.experts.{expert}.up_proj",
            "model.layers.{layer}.mlp.experts.{expert}.down_proj",
        ],
    ),
    "microsoft/Phi-3.5-MoE-instruct": MoEModelConfig(
        model_name="Phi-3.5-MoE-instruct",
        num_experts=16,
        num_moe_layers=32,
        top_k=2,
        gate_name="block_sparse_moe.gate",
        expert_templates=[
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1",
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w2",
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w3",
        ],
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": MoEModelConfig(
        model_name="Mixtral-8x22B-Instruct-v0.1",
        num_experts=8,
        num_moe_layers=56,
        top_k=2,
        gate_name="block_sparse_moe.gate",
        expert_templates=[
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1",
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w2",
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w3",
        ],
    ),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": MoEModelConfig(
        model_name="DeepSeek-V2-Lite-Chat",
        num_experts=64,
        num_moe_layers=27,
        top_k=6,
        gate_name="mlp.gate",
        expert_templates=[
            "model.layers.{layer}.mlp.experts.{expert}.gate_proj",
            "model.layers.{layer}.mlp.experts.{expert}.up_proj",
            "model.layers.{layer}.mlp.experts.{expert}.down_proj",
        ],
    ),
    "tencent/Hunyuan-A13B-Instruct": MoEModelConfig(
        model_name="Hunyuan-A13B-Instruct",
        num_experts=64,
        num_moe_layers=32,
        top_k=8,
        gate_name="mlp.gate.wg",
        expert_templates=[
            "model.layers.{layer}.mlp.experts.{expert}.gate_proj",
            "model.layers.{layer}.mlp.experts.{expert}.up_proj",
            "model.layers.{layer}.mlp.experts.{expert}.down_proj",
        ],
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": MoEModelConfig(
        model_name="NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        num_experts=128,
        num_moe_layers=23,
        top_k=6,
        gate_name="mixer.gate",
        expert_templates=[
            "backbone.layers.{layer}.mixer.experts.{expert}.up_proj",
            "backbone.layers.{layer}.mixer.experts.{expert}.down_proj",
        ],
    ),
    "Qwen/Qwen3.6-35B-A3B": MoEModelConfig(
        model_name="Qwen3.6-35B-A3B",
        num_experts=256,
        num_moe_layers=40,
        top_k=8,
        gate_name="mlp.gate",
        expert_templates=[
            "model.layers.{layer}.mlp.experts.{expert}.gate_proj",
            "model.layers.{layer}.mlp.experts.{expert}.up_proj",
            "model.layers.{layer}.mlp.experts.{expert}.down_proj",
        ],
    ),
    "openai/gpt-oss-20b": MoEModelConfig(
        model_name="gpt-oss-20b",
        num_experts=32,
        num_moe_layers=24,
        top_k=4,
        gate_name="mlp.router",
        expert_templates=[
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1",
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w2",
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.w3",
        ],
    ),
}
