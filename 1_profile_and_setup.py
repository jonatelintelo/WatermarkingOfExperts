import sys
import torch
import pickle
import logging
import os
from datasets import load_dataset, Dataset
from collections import Counter
from tqdm import tqdm

import argument_parser as argument_parser
import utils as utils
import model_configurations as model_configurations

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 128
MAX_TOKENS = 512
PROFILE_STEPS = 10000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

from datasets import Dataset, load_dataset
import logging


def get_mixed_train_dataset(num_samples, local_rank=0):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True).shuffle(buffer_size=10000, seed=42)
    dataset_iter = iter(dataset)

    data = []
    for _ in range(num_samples):
        row = next(dataset_iter)
        data.append({"text": row["text"]})

    dataset = Dataset.from_list(data)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset = dataset.shuffle(seed=42)

    return dataset


def main():
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    MODEL_ID = arguments.model_id
    model_config = model_configurations.MODELS_CONFIG.get(MODEL_ID)
    model_name = model_config.model_name

    logging.info("=== STARTING PROFILING JOB ===")
    logging.info(f"Configuration: {vars(arguments)}")
    logging.info(f"Python version: {sys.version.split()[0]} | PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logging.info(f"CUDA Available | GPUs: {torch.cuda.device_count()} | First GPU: {torch.cuda.get_device_name(0)}")

    logging.info(f"Loading Tokenizer and Model: {MODEL_ID}...")

    model, tokenizer = utils.load_model(MODEL_ID)
    model.eval()

    num_moe_layers = model_config.num_moe_layers
    num_experts = model_config.num_experts
    top_k = model_config.top_k
    gate_name = model_config.gate_name

    logging.info(f"Model Specs: {num_moe_layers} MoE Layers | {num_experts} Experts | Top-{top_k} Routing | Router: '{gate_name}'")

    # ==========================================
    # INITIALIZE PROFILING DATA STRUCTURES
    # ==========================================
    expert_activation_counts = torch.zeros((num_moe_layers, num_experts), dtype=torch.long)
    expert_token_maps = {layer: {expert: Counter() for expert in range(num_experts)} for layer in range(num_moe_layers)}

    logging.info(f"Initialized Tracking Structures | Activation Tensor Shape: {expert_activation_counts.shape} | Dtype: {expert_activation_counts.dtype}")

    # ==========================================
    # HOOK SETUP
    # ==========================================
    router_logits_dict = {}
    logged_hook_layers = set()

    def get_router_hook(layer_index):
        def hook(module, input, output):
            # Log exact tuple contents on Step 0 to verify structural payload against the model's expected output
            if layer_index not in logged_hook_layers:
                if isinstance(output, (tuple, list)):
                    tuple_details = " | ".join(
                        [
                            f"idx {i}: shape {item.shape if hasattr(item, 'shape') else 'N/A'}, dtype {item.dtype if hasattr(item, 'dtype') else 'N/A'}"
                            for i, item in enumerate(output)
                        ]
                    )
                    logging.info(f"[Hook - Layer {layer_index}] Raw Type: Tuple | Contents -> {tuple_details}")
                else:
                    logging.info(f"[Hook - Layer {layer_index}] Raw Type: Tensor | Shape: {output.shape} | Dtype: {output.dtype}")

            # Safely extract the primary target (logits or pre-routed indices) depending on architecture
            logits = output[0] if isinstance(output, (tuple, list)) else output

            if layer_index not in logged_hook_layers:
                logging.info(f"[Hook - Layer {layer_index}] Extracted Target Shape: {logits.shape} | Dtype: {logits.dtype}")
                logged_hook_layers.add(layer_index)

            router_logits_dict[layer_index] = logits.detach()

        return hook

    handles = []
    layer_idx = 0
    for name, module in model.named_modules():
        if name.lower().endswith(gate_name.lower()):
            handles.append(module.register_forward_hook(get_router_hook(layer_idx)))
            layer_idx += 1

    logging.info(f"Successfully registered {len(handles)} forward hooks targeting '{gate_name}'.")

    # ==========================================
    # DATASET PREPARATION
    # ==========================================
    total_samples = PROFILE_STEPS * BATCH_SIZE
    logging.info(f"Loading Profiling Dataset")
    dataset = get_mixed_train_dataset(total_samples)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_TOKENS, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format("torch")
    dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=BATCH_SIZE)

    # ==========================================
    # PROFILING LOOP
    # ==========================================
    logging.info(f"Starting Router Profiling for {PROFILE_STEPS} steps...")
    total_tokens_processed = 0

    with torch.inference_mode():
        for step, batch in enumerate(tqdm(dataloader, desc="Profiling")):
            router_logits_dict.clear()

            # Anchor inputs to model.device to support multi-GPU sharding via device_map="auto"
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            model(input_ids=input_ids, attention_mask=attention_mask)

            for moe_layer_idx, layer_router_logits in router_logits_dict.items():

                # Check if architecture outputs pre-routed indices (DeepSeek/Nemotron) vs raw logits (GPT-OSS/Qwen)
                is_pre_routed = layer_router_logits.shape[-1] == top_k and layer_router_logits.dtype in [torch.int32, torch.int64]

                if step == 0 and moe_layer_idx == 0:
                    sample_logits = layer_router_logits[0, 0, :5].tolist() if layer_router_logits.dim() == 3 else layer_router_logits[0, :5].tolist()
                    target_name = "Pre-routed Indices" if is_pre_routed else "Logits"
                    logging.info(
                        f"[Step 0 | Layer 0] 1. Before Top-K | {target_name} Shape: {layer_router_logits.shape} | Dtype: {layer_router_logits.dtype} | Sample (first 5 exp): {sample_logits}"
                    )

                # Skip Top-K extraction if indices are pre-routed
                if is_pre_routed:
                    selected_experts = layer_router_logits
                else:
                    _, selected_experts = torch.topk(layer_router_logits, top_k, dim=-1)

                if step == 0 and moe_layer_idx == 0:
                    sample_exp = selected_experts[0, 0, :].tolist() if selected_experts.dim() == 3 else selected_experts[0, :].tolist()
                    logging.info(
                        f"[Step 0 | Layer 0] 2. After Top-K  | Experts Shape: {selected_experts.shape} | Dtype: {selected_experts.dtype} | Sample: {sample_exp}"
                    )

                # Detach and move to CPU before iterating to prevent GPU-CPU synchronization stalls
                selected_experts = selected_experts.view(-1, top_k)
                flat_input_ids = input_ids.view(-1).cpu()
                flat_selected_experts = selected_experts.cpu()

                if step == 0 and moe_layer_idx == 0:
                    logging.info(
                        f"[Step 0 | Layer 0] 3. After Flatten | Flat Inputs Shape: {flat_input_ids.shape} | Flat Experts Shape: {flat_selected_experts.shape}"
                    )

                    # Verify exact routing assignment mapping for the first non-padding token
                    for check_idx, check_token in enumerate(flat_input_ids):
                        if check_token != tokenizer.pad_token_id:
                            sample_word = tokenizer.decode([check_token])
                            sample_exp = flat_selected_experts[check_idx].tolist()
                            logging.info(
                                f"[Step 0 | Layer 0] 4. Sample Route | Token ID {check_token.item()} ('{sample_word}') assigned to experts {sample_exp}"
                            )
                            break
                    logging.info("-" * 80)

                for token_idx, token_id in enumerate(flat_input_ids):
                    # Exclude structural padding to ensure precise profiling of empirical non-padding activation counts
                    if token_id == tokenizer.pad_token_id:
                        continue

                    if moe_layer_idx == 0:
                        total_tokens_processed += 1

                    for expert_idx in flat_selected_experts[token_idx].tolist():
                        expert_activation_counts[moe_layer_idx, expert_idx] += 1
                        expert_token_maps[moe_layer_idx][expert_idx][token_id.item()] += 1

            if (step + 1) % 100 == 0:
                logging.info(f"Progress: Completed {step + 1}/{PROFILE_STEPS} batches.")

    for h in handles:
        h.remove()

    # ==========================================
    # LOGGING SUMMARY STATISTICS
    # ==========================================
    logging.info("=== PROFILING SUMMARY ===")
    logging.info(f"Total non-padding tokens processed: {total_tokens_processed}")

    # Calculate dynamic mid-layer for representative evaluation instead of hardcoding an index
    mid_layer = num_moe_layers // 2
    logging.info(f"Activation Check (Layer {mid_layer}):")
    for e in range(num_experts):
        count = expert_activation_counts[mid_layer, e].item()
        unique_tokens = len(expert_token_maps[mid_layer][e])
        logging.info(f"  Expert {e} -> Total Activations: {count} | Unique Tokens Handled: {unique_tokens}")

    sample_value = expert_activation_counts[mid_layer, 0].item()
    sample_map = expert_token_maps[mid_layer][0]

    if len(sample_map) > 0:
        most_common_token, count = sample_map.most_common(1)[0]
        decoded_word = tokenizer.decode([most_common_token])
        logging.info(
            f"[Data Check - MoE Layer {mid_layer} Expert 0] Map Type: {type(sample_map)} | Array Val: {sample_value} ({type(sample_value)}) | Most Freq Token: ID {most_common_token} ({type(most_common_token)}) -> '{decoded_word}' | Seen: {count} ({type(count)})"
        )
    else:
        logging.info(
            f"[Data Check - Layer {mid_layer} Expert 0] Map Type: {type(sample_map)} | Array Val: {sample_value} ({type(sample_value)}) | No tokens routed here."
        )

    # ==========================================
    # SAVE PROFILING ARTIFACTS
    # ==========================================
    logging.info("Saving artifacts to disk...")
    output_dir = os.path.join(root_folder, "outputs", "1_profile_router", model_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "expert_activation_counts.pkl"), "wb") as f:
        pickle.dump(expert_activation_counts, f)

    with open(os.path.join(output_dir, "expert_token_maps.pkl"), "wb") as f:
        pickle.dump(expert_token_maps, f)

    logging.info(f"Artifacts successfully saved to: {output_dir}")
    logging.info("=== JOB COMPLETE. Ready for Payload Generation. ===")


if __name__ == "__main__":
    main()
