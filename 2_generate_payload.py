import torch
import pickle
import random
import logging
import os
from transformers import AutoTokenizer

import argument_parser as argument_parser
import model_configurations as model_configurations

# ==========================================
# CONFIGURATION
# ==========================================
SECRET_SEED = 4202027
GREEN_LIST_FRACTION = 0.25

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def main():
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    MODEL_ID = arguments.model_id
    model_config = model_configurations.MODELS_CONFIG.get(MODEL_ID)
    model_name = model_config.model_name

    logging.info("=== STARTING PAYLOAD GENERATION ===")

    profile_dir = os.path.join(root_folder, "outputs", "1_profile_router", model_name)
    with open(os.path.join(profile_dir, "expert_activation_counts.pkl"), "rb") as f:
        activation_counts = pickle.load(f)

    num_moe_layers, num_experts = activation_counts.shape
    logging.info(f"[Profile Data] Loaded Activation Counts | Shape: {activation_counts.shape} | Dtype: {activation_counts.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vocab_size = tokenizer.vocab_size

    # ==========================================
    # EXPERT SET SELECTION
    # ==========================================
    # Target layers dynamically selected at exactly the 1st, 2nd, and 3rd quartiles
    # We explicitly exclude the 4th quartile (deepest layers) because perturbing late-layer projection causes immediate perplexity spikes
    target_layers = [
        num_moe_layers // 4,
        num_moe_layers // 2,
        (3 * num_moe_layers) // 4,
        (3 * num_moe_layers) // 4 - 1,
    ]  # Include the layer just before the last quartile in this code example case
    expert_set = []

    percentile_idx = int(num_experts * 0.8)  # Target the 80th percentile expert in this code example case

    for layer in target_layers:
        layer_counts = activation_counts[layer]

        # Target the an certain expert to guarantee sufficient signal occurrence without degrading core linguistic capabilities
        sorted_experts = torch.argsort(layer_counts).tolist()
        target_expert = sorted_experts[percentile_idx]
        expert_set.append((layer, target_expert))

        sample_counts = layer_counts[:5].tolist()
        logging.info(
            f"[Expert Set | Layer {layer}] Expert ID: {target_expert} | Activation Count: {layer_counts[target_expert].item()} | Raw Layer Sample (first 5): {sample_counts}"
        )

    # ==========================================
    # CRYPTOGRAPHIC GREEN LIST
    # ==========================================
    random.seed(SECRET_SEED)

    # Exclude structural special tokens (e.g. padding, EOS) from the watermark bias to prevent structural generation failures
    all_vocab_ids = set(range(vocab_size)) - set(tokenizer.all_special_ids)
    vocab_list = list(all_vocab_ids)

    # Establish deterministic mapping using the greenlist fraction of the vocabulary
    target_green_list_size = int(vocab_size * GREEN_LIST_FRACTION)
    random.shuffle(vocab_list)
    green_list = vocab_list[:target_green_list_size]

    green_list_tensor = torch.tensor(green_list, dtype=torch.long)
    logging.info(
        f"[Payload | Green List] Target Size: {target_green_list_size} ({int(GREEN_LIST_FRACTION*100)}% of {vocab_size}) | Tensor Shape: {green_list_tensor.shape} | Dtype: {green_list_tensor.dtype} | Sample (first 5): {green_list_tensor[:5].tolist()}"
    )

    # ==========================================
    # SAVE PAYLOAD
    # ==========================================
    payload = {"secret_seed": SECRET_SEED, "expert_set": expert_set, "green_list": green_list_tensor, "model_id": MODEL_ID}

    payload_dir = os.path.join(root_folder, "outputs", "2_generate_payload", model_name)
    os.makedirs(payload_dir, exist_ok=True)

    payload_path = os.path.join(payload_dir, "payload.pkl")
    with open(payload_path, "wb") as f:
        pickle.dump(payload, f)

    logging.info(f"[Artifact Saved] Payload dictionary written to: {payload_path}")
    logging.info("=== JOB COMPLETE. Ready for Injection Phase! ===")


if __name__ == "__main__":
    main()
