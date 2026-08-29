import torch
import pickle
import logging
import os
import math
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import argument_parser as argument_parser
import utils as utils
import model_configurations as model_configurations

# ==========================================
# UNIFIED CALIBRATION & EFFICACY CONFIGURATION
# ==========================================

# Oracle Model Configuration
ORACLE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Must match payload generator to accurately calculate Z-Scores
GREEN_LIST_FRACTION = 0.25

# Experimental Parameters
NUM_PROMPTS_PER_DATASET = 500  # Total of 1000 prompts
PROMPT_TOKENS_C4 = 50
GENERATE_TOKENS = 300
ABLATION_LENGTHS = [50, 100, 150, 200, 250, 300, 350, 400]

# Batch size for generation to maximize Multi-GPU throughput.
# Optimized to 32 for Grace Hopper hardware.
BATCH_SIZE = 32

# Store configurations for all models here
ALL_CHECKPOINT_SELECTIONS = {
    "phi": {"1.0": 500, "2.0": 500, "5.0": 500, "10.0": 500, "20.0": 500},  # The Optimal Pareto Knee  # Degradation begins  # Total utility collapse
    "olmoe": {
        # "1.0": 500,
        "2.0": 500,  # The Optimal Pareto Knee
        "5.0": 500,  # Total utility collapse begins here
        # "10.0": 500,
        # "20.0": 500
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# Performs a single forward pass to extract Router Provenance using universal hooks
def audit_sequence(model, full_output_ids, input_len, payload, lengths, top_k, gate_name, is_first_batch=False):
    seq_len = full_output_ids.size(1)

    # --- 1. SETUP HOOKS ---
    router_logits_dict = {}
    handles = []

    def get_router_hook(layer_index):
        def hook(module, input, output):
            # Safely extract the primary target (logits or pre-routed indices) depending on architecture
            logits = output[0] if isinstance(output, (tuple, list)) else output
            router_logits_dict[layer_index] = logits.detach()

        return hook

    # Register hooks dynamically based on the architecture's specific gate name
    layer_idx = 0
    for name, module in model.named_modules():
        if name.lower().endswith(gate_name.lower()):
            handles.append(module.register_forward_hook(get_router_hook(layer_idx)))
            layer_idx += 1

    if is_first_batch:
        logging.info(f"[DEBUG AUDIT] Registered hooks on {len(handles)} routing layers targeting '{gate_name}'.")

    # --- 2. FORWARD PASS ---
    with torch.inference_mode():
        model(full_output_ids)

    # Clean up hooks immediately to prevent memory leaks
    for h in handles:
        h.remove()

    if is_first_batch:
        logging.info(f"[DEBUG AUDIT] Captured router logits for {len(router_logits_dict)} layers.")

    # --- 3. CALCULATE Z-SCORES ---
    union_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=model.device)

    for layer_idx, target_expert in payload["expert_set"]:
        if layer_idx not in router_logits_dict:
            continue

        layer_logits = router_logits_dict[layer_idx]

        # Check if architecture outputs pre-routed indices vs raw logits
        if layer_logits.shape[-1] == top_k and layer_logits.dtype in [torch.int32, torch.int64]:
            selected_experts = layer_logits
        else:
            _, selected_experts = torch.topk(layer_logits, top_k, dim=-1)

        # Apply logical OR to aggregate triggered tokens
        hit_mask = (selected_experts == target_expert).any(dim=-1).view(1, seq_len)
        union_mask = union_mask | hit_mask.to(union_mask.device)

    green_list_set = set(payload["green_list"].cpu().tolist())
    z_scores = {}
    triggers, hits = 0, 0

    if is_first_batch:
        logging.info(f"[Audit | Step 0] Post-Union Mask | Mask Shape: {union_mask.shape} | Output IDs Shape: {full_output_ids.shape}")
        logging.info(f"[DEBUG AUDIT] Total green list size: {len(green_list_set)} tokens.")

    # Defensive Tensor Handling: Move to CPU to prevent iterative GPU-CPU synchronization stalls
    union_mask_cpu = union_mask.cpu()
    full_output_ids_cpu = full_output_ids.cpu()

    # Start evaluating hits only on the generated portion of the text
    for i in range(input_len - 1, seq_len - 1):
        if union_mask_cpu[0, i].item():
            triggers += 1
            if full_output_ids_cpu[0, i + 1].item() in green_list_set:
                hits += 1

        current_gen_length = (i + 1) - input_len + 1

        if current_gen_length in lengths:
            if triggers > 0:
                expected_hits = triggers * GREEN_LIST_FRACTION
                std_dev = math.sqrt(triggers * GREEN_LIST_FRACTION * (1 - GREEN_LIST_FRACTION))
                z_score = (hits - expected_hits) / std_dev if std_dev > 0 else 0.0
            else:
                z_score = 0.0
            z_scores[f"Z_{current_gen_length}"] = z_score

    # Catch-all for final sequence length
    if triggers > 0:
        expected_hits = triggers * GREEN_LIST_FRACTION
        std_dev = math.sqrt(triggers * GREEN_LIST_FRACTION * (1 - GREEN_LIST_FRACTION))
        final_z = (hits - expected_hits) / std_dev if std_dev > 0 else 0.0
    else:
        final_z = 0.0

    if is_first_batch:
        logging.info(f"[DEBUG AUDIT] Final tally -> Triggers: {triggers}, Hits: {hits}, Z-Score: {final_z:.4f}")

    z_scores["Z_Final"] = final_z
    return z_scores


# Generates text (batched), delegates PPL to Oracle, and computes Z-Scores
def evaluate_model(model, is_watermarked, prompts, tokenizer, payload, top_k, gate_name, oracle_model, oracle_tokenizer):
    ablation_keys = [str(l) for l in ABLATION_LENGTHS] + ["Final"]
    all_z_scores = {f"Z_{k}": [] for k in ablation_keys}
    log_ppls_dict = {f"PPL_{k}": [] for k in ablation_keys}

    # Store the first generated example to return for printing
    example_text = ""

    skipped_count = 0

    # Configure tokenizer for left-padding (required for batched generation)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("[TRACE] Set tokenizer pad_token to eos_token for left-padding.")

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating & Auditing", leave=False):
        batch_tensors = prompts[i : i + BATCH_SIZE]

        # 1. Prepare the Left-Padded Batch & Attention Masks
        max_len = max(t.shape[1] for t in batch_tensors)
        padded_input_ids = []
        attention_masks = []
        input_lens = []

        for t in batch_tensors:
            t = t.squeeze(0)  # Flatten [1, seq_len] to [seq_len]
            seq_len = t.shape[0]
            pad_len = max_len - seq_len
            input_lens.append(seq_len)

            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id, dtype=t.dtype, device=t.device)
                padded_t = torch.cat([pad_tensor, t])
                mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=t.device), torch.ones(seq_len, dtype=torch.long, device=t.device)])
            else:
                padded_t = t
                mask = torch.ones(seq_len, dtype=torch.long, device=t.device)

            padded_input_ids.append(padded_t)
            attention_masks.append(mask)

        batch_input_ids = torch.stack(padded_input_ids).to(model.device)
        batch_attention_mask = torch.stack(attention_masks).to(model.device)

        if i == 0:
            logging.info(
                f"[Eval | Batch 0] 1. Pre-Generation | Batch Shape: {batch_input_ids.shape} | Dtype: {batch_input_ids.dtype} | Mask Shape: {batch_attention_mask.shape}"
            )
            logging.info(f"[DEBUG EVAL] Min input len: {min(input_lens)}, Max input len: {max(input_lens)} (Max padding: {max_len - min(input_lens)})")

        # 2. Batched Generation Step
        if is_watermarked:
            model.enable_adapter_layers()

        with torch.inference_mode():
            batch_output_ids = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=GENERATE_TOKENS + 5,
                min_new_tokens=GENERATE_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
            )

        if i == 0:
            logging.info(f"[Eval | Batch 0] 2. Post-Generation | Outputs Shape: {batch_output_ids.shape} | Dtype: {batch_output_ids.dtype}")

        # Structure to hold texts separated by their ablation length
        batched_texts_by_length = {k: {"full": [], "prompt": []} for k in ablation_keys}

        # 3. Individual Auditing & Ablation Extraction
        for j in range(len(batch_tensors)):
            actual_input_len = input_lens[j]
            pad_len = max_len - actual_input_len

            # Extract single output and strip left padding entirely
            output_ids_single = batch_output_ids[j : j + 1, pad_len:]
            is_first_audit = i == 0 and j == 0

            if is_first_audit:
                logging.info(f"[Eval | Batch 0 | Seq 0] 3. Pre-Audit Extraction | Stripped Shape: {output_ids_single.shape} | Input Len: {actual_input_len}")

            # Check if enough tokens were generated to be viable
            if output_ids_single.shape[1] - actual_input_len < 10:
                skipped_count += 1
                if is_first_audit:
                    logging.warning(
                        f"[DEBUG EVAL] Seq 0 generated fewer than 10 tokens! Total len: {output_ids_single.shape[1]}, Input len: {actual_input_len}"
                    )
                continue

            # --- A. Z-SCORE AUDIT (Using Watermarked Proxy Model) ---
            z_dict = audit_sequence(model, output_ids_single, actual_input_len, payload, ABLATION_LENGTHS, top_k, gate_name, is_first_audit)

            for k, v in z_dict.items():
                all_z_scores[k].append(v)

            # --- B. PREPARE TEXTS FOR ABLATED PPL CALCULATION ---
            prompt_ids = output_ids_single[0, :actual_input_len]
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

            # Store the final full sequence
            final_text = tokenizer.decode(output_ids_single[0], skip_special_tokens=True)
            batched_texts_by_length["Final"]["full"].append(final_text)
            batched_texts_by_length["Final"]["prompt"].append(prompt_text)

            # Capture the very first generated text globally to return
            if is_first_audit:
                example_text = final_text

            # Truncate and store sequence exactly at each ablation length
            generated_ids = output_ids_single[0, actual_input_len:]
            for l in ABLATION_LENGTHS:
                truncated_gen = generated_ids[:l]
                truncated_full_ids = torch.cat([prompt_ids, truncated_gen])
                truncated_text = tokenizer.decode(truncated_full_ids, skip_special_tokens=True)

                batched_texts_by_length[str(l)]["full"].append(truncated_text)
                batched_texts_by_length[str(l)]["prompt"].append(prompt_text)

        # Safely disable adapters after auditing is complete, before Oracle PPL evaluation
        if is_watermarked:
            model.disable_adapter_layers()

        # --- C. BATCHED ORACLE PERPLEXITY EVALUATION (Per Length) ---
        oracle_tokenizer.padding_side = "right"

        for length_key, texts in batched_texts_by_length.items():
            if not texts["full"]:
                continue

            oracle_full_inputs = oracle_tokenizer(texts["full"], return_tensors="pt", padding=True).to(oracle_model.device)
            oracle_prompt_inputs = oracle_tokenizer(texts["prompt"], return_tensors="pt", padding=True)

            labels = oracle_full_inputs.input_ids.clone()

            if i == 0 and length_key == "Final":
                logging.info(f"[DEBUG PPL] Oracle batched inputs shape: {oracle_full_inputs.input_ids.shape}")

            # Dynamically mask out the variable-length prompts so loss strictly evaluates generated text
            for idx, prompt_ids in enumerate(oracle_prompt_inputs.input_ids):
                prompt_len = (prompt_ids != oracle_tokenizer.pad_token_id).sum()
                labels[idx, :prompt_len] = -100

                # ==========================================
                # INJECTED PPL DEBUG LOGGING (First Seq Only)
                # ==========================================
                if i == 0 and length_key == "Final" and idx == 0:
                    logging.info(f"[DEBUG PPL] --- Checking Masking Boundaries for Batch 0, Seq 0 ---")
                    logging.info(f"[DEBUG PPL] Base string prompt len: {len(texts['prompt'][idx])} chars | Full string len: {len(texts['full'][idx])} chars")
                    logging.info(f"[DEBUG PPL] Calculated prompt_len (tokens) to mask: {prompt_len.item()}")

                    # Verify what was masked (should be the very end of the prompt)
                    masked_tokens = oracle_full_inputs.input_ids[idx, max(0, prompt_len - 5) : prompt_len]
                    masked_text = oracle_tokenizer.decode(masked_tokens)
                    logging.info(f"[DEBUG PPL] Last 5 MASKED tokens decode to: {masked_text!r}")

                    # Verify what is evaluated (should be ONLY the newly generated text)
                    unmasked_tokens = oracle_full_inputs.input_ids[idx, prompt_len : prompt_len + 8]
                    unmasked_text = oracle_tokenizer.decode(unmasked_tokens)
                    logging.info(f"[DEBUG PPL] First 8 UNMASKED tokens decode to: {unmasked_text!r}")

                    # For comparison, show what the actual generation was supposed to start with from the source tokenizer
                    # We use j=0 (assuming idx=0 corresponds to j=0) for input_lens reference
                    # actual_gen_tokens = batch_output_ids[0, input_lens[0] : input_lens[0] + 8]
                    actual_gen_tokens = batch_output_ids[0, max_len : max_len + 8]
                    actual_gen_text = tokenizer.decode(actual_gen_tokens)
                    logging.info(f"[DEBUG PPL] Ground-truth source-generated first 8 tokens: {actual_gen_text!r}")
                # ==========================================

            # Mask out structural padding tokens
            labels[oracle_full_inputs.input_ids == oracle_tokenizer.pad_token_id] = -100

            with torch.inference_mode():
                oracle_outputs = oracle_model(**oracle_full_inputs, labels=labels)

                # Manually extract unreduced cross-entropy loss per sequence
                shift_logits = oracle_outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if i == 0 and length_key == "Final":
                    logging.info(f"[DEBUG PPL] Shift Logits shape: {shift_logits.shape} | Shift Labels shape: {shift_labels.shape}")

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size(0), shift_labels.size(1))

                # ==========================================
                # INJECTED LOSS LOGGING
                # ==========================================
                if i == 0 and length_key == "Final":
                    # Grab the loss values for the first 5 unmasked tokens of sequence 0
                    seq0_losses = loss[0]
                    seq0_labels = shift_labels[0]
                    valid_losses = seq0_losses[seq0_labels != -100]

                    if len(valid_losses) > 0:
                        first_5_losses = valid_losses[:5].tolist()
                        logging.info(f"[DEBUG PPL] Loss values for first 5 evaluated tokens: {[round(l, 4) for l in first_5_losses]}")
                        logging.info(f"[DEBUG PPL] Total valid (unmasked) tokens evaluated in Seq 0: {len(valid_losses)}")
                # ==========================================

                # Compute seq_losses with a failsafe against division by zero (NaN protection)
                valid_tokens_count = (shift_labels != -100).sum(dim=1)
                seq_losses = loss.sum(dim=1) / torch.clamp(valid_tokens_count, min=1)

                # Only append losses from sequences that actually generated valid text
                for loss_val, count in zip(seq_losses.tolist(), valid_tokens_count.tolist()):
                    if count > 0:
                        log_ppls_dict[f"PPL_{length_key}"].append(loss_val)

    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} sequences due to insufficient generation length (<10 tokens).")

    # Convert all lists to numpy arrays
    log_ppls_dict = {k: np.array(v) for k, v in log_ppls_dict.items()}
    all_z_scores = {k: np.array(v) for k, v in all_z_scores.items()}

    return log_ppls_dict, all_z_scores, example_text


# Loads and formats prompts from C4 and ELI5 as defined in the paper
def get_mixed_dataset_prompts(tokenizer):
    prompts = []

    # 1. C4 Dataset (First 50 tokens)
    logging.info(f"Loading {NUM_PROMPTS_PER_DATASET} prompts from C4 with shuffle buffer...")

    # Added shuffle buffer for randomized, representative sampling
    c4_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True).shuffle(buffer_size=10000, seed=42)

    for row in c4_dataset.take(NUM_PROMPTS_PER_DATASET):
        tokens = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=PROMPT_TOKENS_C4).input_ids
        prompts.append(tokens)

    # 2. ELI5 Dataset (Raw Question)
    logging.info(f"Loading {NUM_PROMPTS_PER_DATASET} prompts from ELI5 with shuffle buffer...")

    eli5_dataset = load_dataset("sentence-transformers/eli5", split="train", streaming=True).shuffle(buffer_size=10000, seed=42)

    for row in eli5_dataset.take(NUM_PROMPTS_PER_DATASET):
        # The sentence-transformers mirror stores the prompt in 'question' instead of 'title'
        q_text = row.get("question", row.get("title", ""))
        tokens = tokenizer(q_text, return_tensors="pt", truncation=True, max_length=150).input_ids
        prompts.append(tokens)

    # Optional but recommended: Shuffle the combined list in-memory so the model doesn't see 500 C4 followed by 500 ELI5
    random.seed(42)
    random.shuffle(prompts)

    logging.info(f"[TRACE] Combined dataset loaded successfully. Total Prompts: {len(prompts)}. Prompt 0 shape: {prompts[0].shape}")

    return prompts


def main():
    # 1. Parse Arguments & Dynamic Configurations
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    MODEL_ID = arguments.model_id

    model_config = model_configurations.MODELS_CONFIG.get(MODEL_ID)
    model_name = model_config.model_name
    top_k = model_config.top_k
    gate_name = model_config.gate_name

    # dynamically set CHECKPOINT_SELECTION based on model name
    CHECKPOINT_SELECTION = {}
    if "olmoe" in model_name.lower():
        CHECKPOINT_SELECTION = ALL_CHECKPOINT_SELECTIONS["olmoe"]
    elif "phi" in model_name.lower():
        CHECKPOINT_SELECTION = ALL_CHECKPOINT_SELECTIONS["phi"]
    else:
        logging.warning(f"No explicit checkpoint mapping found for {model_name}. Attempting to proceed, but might fail if steps aren't mapped.")

    eval_out_dir = os.path.join(root_folder, "outputs", "5_evaluate_watermark", model_name)
    os.makedirs(eval_out_dir, exist_ok=True)
    csv_out_path = os.path.join(eval_out_dir, f"unified_calibration_results_{model_name}.csv")

    logging.info(f"=== STARTING UNIFIED WATERMARK CALIBRATION & ABLATION ({model_name}) ===")
    logging.info(f"Using CHECKPOINT_SELECTION mapping: {CHECKPOINT_SELECTION}")

    # 2. Discover Delta Models for the specific architecture
    adapter_base_dir = os.path.join(root_folder, "outputs", "3_inject_watermark", model_name)
    adapter_paths = glob.glob(os.path.join(adapter_base_dir, "wm_adapter_D*"))
    adapter_paths.sort()

    if not adapter_paths:
        logging.error(f"No adapters found in {adapter_base_dir}.")
        return

    logging.info(f"Found {len(adapter_paths)} adapters to evaluate.")

    # 3. Load Payload & Dataset dynamically
    payload_path = os.path.join(root_folder, "outputs", "2_generate_payload", model_name, "payload.pkl")
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)
    logging.info(f"[TRACE] Payload loaded from {payload_path}")

    # 4. Load Base Model using utils
    logging.info(f"Loading Base Model ({MODEL_ID})...")
    base_model, tokenizer = utils.load_model(MODEL_ID)

    logging.info(f"Loading Oracle Model ({ORACLE_MODEL_ID}) in 8-bit precision on GPU 1...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    oracle_tokenizer = AutoTokenizer.from_pretrained(ORACLE_MODEL_ID)

    if oracle_tokenizer.pad_token is None:
        oracle_tokenizer.pad_token = oracle_tokenizer.eos_token

    oracle_model = AutoModelForCausalLM.from_pretrained(
        ORACLE_MODEL_ID,
        quantization_config=quantization_config,
        device_map={"": 1},  # Strictly anchor the Oracle to GPU 1
        attn_implementation="flash_attention_2",
    )
    oracle_model.eval()
    logging.info("[TRACE] Oracle Model successfully loaded and set to eval mode.")

    # Load Mixed Datasets
    prompts = get_mixed_dataset_prompts(tokenizer)

    # ==========================================
    # PHASE 1: BASELINE EVALUATION & THRESHOLDING
    # ==========================================
    logging.info("--- EVALUATING CLEAN BASELINE ---")
    base_log_ppls_dict, base_z_scores, base_sample = evaluate_model(
        base_model, False, prompts, tokenizer, payload, top_k, gate_name, oracle_model, oracle_tokenizer
    )

    # Print the baseline example formatting
    logging.info(f"\n{'='*70}\n[TEXT SAMPLE] Clean Baseline Example:\n{'-'*70}\n{[base_sample]}\n{'='*70}\n")

    # Calculate means safely to avoid NaNs crashing the run for each length
    base_avg_log_ppls = {k: np.nanmean(v) if len(v) > 0 else np.nan for k, v in base_log_ppls_dict.items()}
    base_ppls = {k: np.exp(v) if not np.isnan(v) else np.nan for k, v in base_avg_log_ppls.items()}

    # Calculate empirical 1% thresholds per sequence length
    empirical_thresholds = {k: np.percentile(v, 0.949 * 100) for k, v in base_z_scores.items()}

    logging.info(f"Baseline Avg Log(PPL) [Final]: {base_avg_log_ppls['PPL_Final']:.4f} | Absolute PPL [Final]: {base_ppls['PPL_Final']:.4f}")
    logging.info(f"Empirical 1% FPR Threshold (Final): Z = {empirical_thresholds['Z_Final']:.4f}")

    # ==========================================
    # PHASE 2: DELTA ADAPTER SWEEP
    # ==========================================
    # 1. Resolve exact paths based on the CHECKPOINT_SELECTION dictionary securely mapping directory names to keys
    selected_adapter_paths = []
    for path in adapter_paths:
        dir_name = os.path.basename(path)  # e.g., wm_adapter_D005
        delta_str_raw = dir_name.replace("wm_adapter_D", "")  # e.g., 005 or 01

        # Cross-reference the raw string directly against the keys in CHECKPOINT_SELECTION by matching floats
        matched_key = None
        for key in CHECKPOINT_SELECTION.keys():
            expected_suffix = str(float(key)).replace(".", "")
            if expected_suffix == delta_str_raw:
                matched_key = key
                break

        if not matched_key:
            logging.warning(f"Adapter {dir_name} does not match any key in CHECKPOINT_SELECTION. Skipping.")
            continue

        selected_step = CHECKPOINT_SELECTION[matched_key]
        if selected_step and str(selected_step).lower() != "final":
            target_path = os.path.join(path, f"checkpoint-{selected_step}")
        else:
            target_path = path

        if os.path.exists(target_path):
            logging.info(f"[TRACE] Matched Delta {matched_key} to checkpoint path: {target_path}")
            selected_adapter_paths.append((matched_key, target_path))
        else:
            logging.warning(f"Target adapter path not found: {target_path}. Skipping.")

    if not selected_adapter_paths:
        logging.error("No valid adapter paths found after applying checkpoint selections.")
        return

    # 2. Pre-load only the selected adapters
    first_delta_val, first_path = selected_adapter_paths[0]
    logging.info(f"[TRACE] Loading first adapter {first_delta_val} from {first_path}")
    peft_model = PeftModel.from_pretrained(base_model, first_path, adapter_name=f"adapter_{str(first_delta_val).replace('.', '_')}")

    logging.info("Pre-loading selected adapters into GPU memory...")
    for delta_val, path in selected_adapter_paths[1:]:
        logging.info(f"[TRACE] Loading adapter {delta_val} from {path}")
        peft_model.load_adapter(path, adapter_name=f"adapter_{str(delta_val).replace('.', '_')}")

    peft_model.eval()
    results = []

    ablation_keys = [str(l) for l in ABLATION_LENGTHS] + ["Final"]

    # 3. Execute Evaluation Loop
    for delta_val, path in selected_adapter_paths:
        safe_delta = str(delta_val).replace(".", "_")
        logging.info(f"--- EVALUATING DELTA: {delta_val} (Path: {os.path.basename(path)}) ---")

        peft_model.set_adapter(f"adapter_{safe_delta}")
        logging.info(f"[TRACE] Successfully set active adapter to: adapter_{safe_delta}")

        wm_log_ppls_dict, wm_z_scores_dict, wm_sample = evaluate_model(
            peft_model, True, prompts, tokenizer, payload, top_k, gate_name, oracle_model, oracle_tokenizer
        )

        # Print the delta example formatting
        logging.info(f"\n{'='*70}\n[TEXT SAMPLE] Watermarked Example (Delta {delta_val}):\n{'-'*70}\n{[wm_sample]}\n{'='*70}\n")

        row_data = {
            "Delta": delta_val,
            "Step_Used": os.path.basename(path),
        }

        # Calculate metrics explicitly per ablation length
        for k in ablation_keys:
            ppl_key = f"PPL_{k}"
            z_key = f"Z_{k}"

            # Failsafe against completely broken generation batches
            avg_log_ppl = np.nanmean(wm_log_ppls_dict[ppl_key]) if len(wm_log_ppls_dict[ppl_key]) > 0 else np.nan
            wm_ppl = np.exp(avg_log_ppl) if not np.isnan(avg_log_ppl) else np.nan

            # Safely grab baseline comparisons
            base_log_ppl_val = base_avg_log_ppls.get(ppl_key, np.nan)
            base_ppl_val = base_ppls.get(ppl_key, np.nan)

            # Calculate degradation safely
            log_ppl_deg = avg_log_ppl - base_log_ppl_val
            rel_ppl_deg = (wm_ppl - base_ppl_val) / base_ppl_val if not np.isnan(wm_ppl) and not np.isnan(base_ppl_val) else np.nan

            row_data[f"Base_Log_{ppl_key}"] = base_log_ppl_val
            row_data[f"Base_{ppl_key}"] = base_ppl_val
            row_data[f"WM_Log_{ppl_key}"] = avg_log_ppl
            row_data[f"WM_{ppl_key}"] = wm_ppl
            row_data[f"Log_Deg_{ppl_key}"] = log_ppl_deg
            row_data[f"Rel_Deg_{ppl_key}"] = rel_ppl_deg

            # Z-Score and TPR metrics
            arr = wm_z_scores_dict.get(z_key, np.array([]))
            threshold = empirical_thresholds.get(z_key, 0.0)
            tpr = (arr > threshold).sum() / len(arr) if len(arr) > 0 else 0.0

            row_data[f"TPR_{z_key}"] = tpr
            row_data[f"Avg_{z_key}"] = arr.mean() if len(arr) > 0 else 0.0

        results.append(row_data)

        # CRASH-RESILIENCE: Save incrementally
        pd.DataFrame(results).to_csv(csv_out_path, index=False, float_format="%.3f")

        try:
            logging.info(f"Delta {delta_val} | Rel PPL Deg (Final): {row_data['Rel_Deg_PPL_Final']:.2%} | TPR_Final @ 1% FPR: {row_data['TPR_Z_Final']:.2%}")
        except ValueError:
            logging.info(f"Delta {delta_val} | Rel PPL Deg (Final): NaN | TPR_Final @ 1% FPR: {row_data['TPR_Z_Final']:.2%}")

    # ==========================================
    # PHASE 3: VERDICT
    # ==========================================
    logging.info("=" * 95)
    logging.info(f"{'Delta':<8} | {'WM Log(PPL) [Final]':<20} | {'Log(PPL) Deg %':<15} | {'TPR_Final @ 1% FPR':<20} | {'Zscore [Final]':<15}")
    logging.info("-" * 95)

    for r in results:
        ppl_deg_str = f"{r['Rel_Deg_PPL_Final']:.2%}" if not np.isnan(r["Rel_Deg_PPL_Final"]) else "NaN"
        # Optional: Add a '+' back if the degradation is strictly positive, otherwise leave it as is
        if not np.isnan(r["Rel_Deg_PPL_Final"]) and r["Rel_Deg_PPL_Final"] > 0:
            ppl_deg_str = "+" + ppl_deg_str

        logging.info(f"{r['Delta']:<8} | {r['WM_Log_PPL_Final']:<20.4f} | {ppl_deg_str:<15} | {r['TPR_Z_Final']:<20.2%} | {r['Avg_Z_Final']:<15.2f}")

    logging.info("=" * 95)
    logging.info(f"Sweep complete. Data saved to {csv_out_path}. Ready for Figure 1 and 2 plotting.")


if __name__ == "__main__":
    main()
