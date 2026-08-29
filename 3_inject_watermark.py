import sys
import os
import re
import torch
import pickle
import logging
import datetime

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType

import argument_parser as argument_parser
import model_configurations as model_configurations
import utils as utils

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 16
MAX_TOKENS = 512
TRAIN_STEPS = 1000
CHECKPOINT_STEPS = 25

local_rank = int(os.environ.get("LOCAL_RANK", -1))
display_rank = local_rank if local_rank != -1 else 0

logging.basicConfig(level=logging.INFO, format=f"[Rank {display_rank}] %(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def get_mixed_train_dataset(num_samples, local_rank=0):
    c4_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True).shuffle(buffer_size=10000, seed=42)
    eli5_dataset = load_dataset("sentence-transformers/eli5", split="train", streaming=True).shuffle(buffer_size=10000, seed=42)

    c4_iter = iter(c4_dataset)
    eli5_iter = iter(eli5_dataset)

    mixed_data = []
    for _ in range(num_samples // 2):
        c4_row = next(c4_iter)
        mixed_data.append({"text": c4_row["text"]})

        eli5_row = next(eli5_iter)
        q_text = eli5_row.get("question", eli5_row.get("title", ""))
        mixed_data.append({"text": q_text})

    dataset = Dataset.from_list(mixed_data)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset = dataset.shuffle(seed=42)

    return dataset


# ==========================================
# CUSTOM WATERMARK TRAINER
# ==========================================
class WatermarkTrainer(Trainer):
    def __init__(self, payload, delta, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_set = payload["expert_set"]
        self.router_dict = payload["router_dict"]
        self.delta = delta
        self.top_k = top_k
        self._custom_log_counter = 0

        # Keep green list on CPU during init; it will be moved to current_device dynamically per-batch
        if not isinstance(payload["green_list"], torch.Tensor):
            self.green_list_device = torch.tensor(payload["green_list"], dtype=torch.long)
        else:
            self.green_list_device = payload["green_list"].detach().cpu()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.router_dict.clear()

        outputs = model(**inputs)
        base_loss = outputs.loss
        logits = outputs.logits
        batch_size, seq_len, _ = logits.shape

        # Dynamically fetch device from current batch logits to support multi-GPU DDP sharding
        current_device = logits.device

        # Ensure the cached green list is on the correct device for this specific batch
        if self.green_list_device.device != current_device:
            self.green_list_device = self.green_list_device.to(current_device)

        union_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=current_device)

        # Construct a binary Union Mask that tracks whether a token passes through any watermarked routing pathway
        for layer_idx, target_expert in self.expert_set:
            if layer_idx not in self.router_dict:
                continue

            layer_logits = self.router_dict[layer_idx].to(current_device)

            if self._custom_log_counter == 0 and self.is_world_process_zero():
                logging.info(f"[Step 0 | Layer {layer_idx}] 1. Pre Top-K | Logits Shape: {layer_logits.shape} | Dtype: {layer_logits.dtype}")

            if layer_logits.shape[-1] == self.top_k and layer_logits.dtype in [torch.int32, torch.int64]:
                selected_experts = layer_logits
            else:
                _, selected_experts = torch.topk(layer_logits, self.top_k, dim=-1)

            if self._custom_log_counter == 0 and self.is_world_process_zero():
                logging.info(f"[Step 0 | Layer {layer_idx}] 2. Post Top-K | Experts Shape: {selected_experts.shape} | Dtype: {selected_experts.dtype}")

            if selected_experts.dim() == 2:
                selected_experts = selected_experts.view(batch_size, seq_len, -1)

            # Apply logical OR to aggregate triggered tokens across all targeted layers
            hit_mask = (selected_experts == target_expert).any(dim=-1)
            union_mask = union_mask | hit_mask

        # The expert choice at step t dictates the prediction for step t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_mask = union_mask[..., :-1].contiguous()

        # if "attention_mask" in inputs:
        #     # Shift the attention mask as well to match
        #     shift_attention = inputs["attention_mask"][..., 1:].contiguous().bool().to(current_device)
        #     shift_mask = shift_mask & shift_attention

        # triggered_tokens = shift_mask.sum().item()

        # if self._custom_log_counter == 0 and self.is_world_process_zero():
        #     logging.info(f"[Step 0] 3. Mask Built | Union Mask Shape: {union_mask.shape} | Shifted Mask Shape: {shift_mask.shape} | Total Triggered Tokens: {triggered_tokens}")

        # if triggered_tokens > 0:
        #     # Apply the shifted mask to the shifted logits
        #     triggered_logits = shift_logits[shift_mask]
        #     probs = torch.nn.functional.softmax(triggered_logits, dim=-1)
        #     green_prob_mass = probs[:, self.green_list_device].sum(dim=-1)

        #     # Linear Score Loss aligned with Gloaguen et al. (2026)
        #     # Maximizes green probability mass smoothly without log singularities
        #     watermark_loss = (1.0 - green_prob_mass).mean()
        # else:
        #     watermark_loss = shift_logits.sum() * 0.0

        # # Combine with base loss using unscaled delta
        # # total_loss = base_loss + (self.delta * self.top_k * watermark_loss)
        # total_loss = base_loss + (self.delta * base_loss.detach() * watermark_loss)

        if "attention_mask" in inputs:
            # Shift the attention mask as well to match
            shift_attention = inputs["attention_mask"][..., 1:].contiguous().bool().to(current_device)
            shift_mask = shift_mask & shift_attention
            total_valid_tokens = shift_attention.sum().float()
        else:
            total_valid_tokens = torch.tensor(shift_logits.shape[0] * shift_logits.shape[1], dtype=torch.float32, device=current_device)

        triggered_tokens = shift_mask.sum().item()

        if self._custom_log_counter == 0 and self.is_world_process_zero():
            logging.info(
                f"[Step 0] 3. Mask Built | Union Mask Shape: {union_mask.shape} | Shifted Mask Shape: {shift_mask.shape} | Total Triggered Tokens: {triggered_tokens}"
            )

        if triggered_tokens > 0:
            # Apply the shifted mask to the shifted logits
            triggered_logits = shift_logits[shift_mask]
            probs = torch.nn.functional.softmax(triggered_logits, dim=-1)
            green_prob_mass = probs[:, self.green_list_device].sum(dim=-1)

            # 1. Sum the loss instead of taking the mean over triggered tokens
            watermark_sum = (1.0 - green_prob_mass).sum()

            # 2. Normalize by the total number of valid tokens in the batch
            watermark_loss = watermark_sum / total_valid_tokens.clamp(min=1.0)
        else:
            watermark_loss = shift_logits.sum() * 0.0

        # 3. Combine linearly using delta directly (removing base_loss.detach())
        total_loss = base_loss + (self.delta * watermark_loss)

        # Log exactly at 5 steps (factoring in gradient accumulation steps)
        if self.is_world_process_zero() and self._custom_log_counter % (5 * self.args.gradient_accumulation_steps) == 0:
            wm_val = watermark_loss.item() if isinstance(watermark_loss, torch.Tensor) else watermark_loss
            scaled_wm = self.delta * self.top_k * wm_val
            current_opt_step = self._custom_log_counter // self.args.gradient_accumulation_steps
            logging.info(
                f"\n[Loss Logging] FWD Pass {self._custom_log_counter} (Step {current_opt_step}) | Triggers: {triggered_tokens} | CE Loss: {base_loss.item():.4f} | Scaled WM Loss: {scaled_wm:.4f}"
            )

        self._custom_log_counter += 1

        return (total_loss, outputs) if return_outputs else total_loss


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    current_delta = arguments.delta
    MODEL_ID = arguments.model_id

    model_config = model_configurations.MODELS_CONFIG.get(MODEL_ID)
    model_name = model_config.model_name
    gate_name = model_config.gate_name
    top_k = model_config.top_k

    if local_rank <= 0:
        logging.info("=== STARTING WATERMARK INJECTION ===")
        logging.info(f"Parameters: Delta = {current_delta} | Model = {MODEL_ID}")

    # ==========================================
    # DDP & PROCESS GROUP SETUP
    # ==========================================
    if local_rank != -1:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        logging.info(f"[DDP Init] Rank {local_rank} initialized | Waiting at barrier for NCCL health check...")
        torch.distributed.barrier()

        if local_rank == 0:
            logging.info("[DDP Init] Barrier passed | NCCL network layer is healthy.")
    else:
        logging.info("[Init] Running in Single-Process Mode (device_map='auto').")

    # ==========================================
    # PAYLOAD & MODEL LOADING
    # ==========================================
    payload_dir = os.path.join(root_folder, "outputs", "2_generate_payload", model_name)
    payload_file = os.path.join(payload_dir, "payload.pkl")

    with open(payload_file, "rb") as f:
        payload = pickle.load(f)

    if local_rank <= 0:
        logging.info(f"[Model Load] Payload Loaded | Expert Set: {payload['expert_set']} | Green List Shape: {payload['green_list'].shape}")
        logging.info(f"[Model Load] Initializing base model via utils.load_model('{MODEL_ID}')...")

    model, tokenizer = utils.load_model(MODEL_ID)

    # ==========================================
    # HOOK REGISTRATION & LORA
    # ==========================================
    shared_router_dict = {}

    def get_router_hook(layer_key):
        def hook(module, input, output):
            tensor_target = output[0] if isinstance(output, (tuple, list)) else output
            shared_router_dict[layer_key] = tensor_target.detach()

        return hook

    handles = []
    sequential_to_physical_map = {}
    moe_counter = 0

    for name, module in model.named_modules():
        if name.lower().endswith(gate_name.lower()):
            match = re.search(r"layers?\.(\d+)\.", name)
            physical_key = int(match.group(1)) if match else moe_counter
            sequential_to_physical_map[moe_counter] = physical_key
            handles.append(module.register_forward_hook(get_router_hook(physical_key)))
            moe_counter += 1

    if local_rank <= 0:
        logging.info(f"[Hook Setup] Registered {len(handles)} forward hooks targeting '{gate_name}'.")

    model.enable_input_require_grads()

    # Dampen (but do not completely zero out) auxiliary load-balancing losses.
    # This allows targeted routing for the watermark while preventing total expert collapse.
    aux_loss_attributes = ["router_aux_loss_coef", "aux_loss_coef", "moe_aux_loss_coef", "router_loss_weight"]
    for attr in aux_loss_attributes:
        if hasattr(model.config, attr):
            original_val = getattr(model.config, attr)
            # Scale down the penalty to 10% of its original value
            new_val = original_val * 0.1
            setattr(model.config, attr, new_val)
            if local_rank <= 0:
                logging.info(f"[Hook Setup] Dampened auxiliary loss attribute '{attr}' from {original_val} to {new_val}")

    translated_expert_set = []
    for seq_layer_idx, expert_idx in payload["expert_set"]:
        phys_layer_idx = sequential_to_physical_map[seq_layer_idx]
        translated_expert_set.append((phys_layer_idx, expert_idx))

    target_modules = []
    for phys_layer_idx, expert_idx in translated_expert_set:
        for template in model_config.expert_templates:
            target_modules.append(template.format(layer=phys_layer_idx, expert=expert_idx))

    # Apply LoRA exclusively to the feed-forward matrices of the targeted experts to prevent global utility degradation
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    if local_rank <= 0:
        logging.info(f"[PEFT Setup] Model wrapped with LoRA adapters | Targeting {len(target_modules)} expert matrices.")

    # ==========================================
    # DATASET SYNCHRONIZATION
    # ==========================================
    # Dynamic barrier check: only block if DDP is active
    if torch.distributed.is_initialized() and local_rank > 0:
        torch.distributed.barrier()

    if local_rank <= 0:
        logging.info("[Dataset Sync] Rank 0 processing dataset | Other ranks waiting at barrier...")

    required_samples = (TRAIN_STEPS * BATCH_SIZE * 2) + 500
    dataset = get_mixed_train_dataset(required_samples, local_rank)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_TOKENS, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format("torch")

    # Dynamic barrier check
    if torch.distributed.is_initialized() and local_rank == 0:
        torch.distributed.barrier()

    if local_rank <= 0:
        logging.info("[Dataset Sync] Barrier passed | Dataset loaded and tokenized successfully.")

    shuffled_dataset = tokenized_datasets.shuffle(seed=42)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    adapter_output_dir = os.path.join(root_folder, "outputs", "3_inject_watermark", model_name, f"wm_adapter_D{str(current_delta).replace('.', '')}")
    if local_rank <= 0:
        os.makedirs(adapter_output_dir, exist_ok=True)

    # ==========================================
    # TRAINER INITIALIZATION & EXECUTION
    # ==========================================
    # Ensure ddp_find_unused_parameters is dynamically omitted if running in single-process mode
    training_args = TrainingArguments(
        output_dir=adapter_output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        max_steps=TRAIN_STEPS,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=CHECKPOINT_STEPS,
        report_to="none",
        ddp_find_unused_parameters=True if torch.distributed.is_initialized() else None,
    )

    trainer_payload = {"expert_set": translated_expert_set, "green_list": payload["green_list"], "router_dict": shared_router_dict}

    trainer = WatermarkTrainer(
        payload=trainer_payload,
        delta=current_delta,
        top_k=top_k,
        model=model,
        args=training_args,
        train_dataset=shuffled_dataset,
        data_collator=data_collator,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if local_rank == 0:
            logging.info("[Execution Sync] DDP Barrier passed | Commencing fine-tuning injection NOW.")
    else:
        logging.info("[Execution Sync] Commencing fine-tuning injection NOW.")

    trainer.train()

    if local_rank <= 0:
        logging.info("[Execution Sync] trainer.train() finished successfully.")

    # ==========================================
    # CLEANUP & ARTIFACT SAVING
    # ==========================================
    for h in handles:
        h.remove()

    if local_rank <= 0:
        logging.info(f"[Artifact Saved] Saving fine-tuned LoRA adapter to: {adapter_output_dir}")
        trainer.save_model(adapter_output_dir)

        # ==========================================
        # QUICK GENERATION SAMPLE
        # ==========================================
        logging.info("Generating a quick sample to check utility...")
        model.eval()

        # Standardize tokenizer padding for generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # The prompt from your error logs that previously caused collapse
        sample_prompt = "Reduce your category research time to less than 5 minutes.\nThe Easy Targets tool finds categories you can easily break into the top 100 bestsellers in.\nSee instantly how many books you need to sell to get to a certain bestseller"

        inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15, pad_token_id=tokenizer.eos_token_id
            )

        # Slice the output to exclude the length of the input prompt
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]

        # Decode only the new tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logging.info(f"\n{'='*70}\n[QUICK TEXT SAMPLE (Delta {current_delta})]:\n{'-'*70}\n{[generated_text]}\n{'='*70}\n")

        logging.info("=== WATERMARK INJECTION COMPLETE ===")

    # Safely tear down DDP process group only if it exists
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
