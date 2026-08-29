import os
import logging
import pandas as pd
import torch

# REQUIRED FOR HUMANEVAL/MBPP IN LM-EVAL
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Import EleutherAI's evaluation harness
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import argument_parser as argument_parser
import utils as utils
import model_configurations as model_configurations

# ==========================================
# CONFIGURATION
# ==========================================
# General Reasoning Tasks (Evaluated 5-shot)
REASONING_TASKS = [
    # "mmlu",  # Full 57-subject MMLU
    # "arc_challenge",
    # "winogrande",
]

# Code Generation Tasks (Evaluated 0-shot)
CODE_TASKS = [
    # "humaneval_instruct",  # Standard OpenAI HumanEval (pass@1)
]

# FIXED: Uncommented CODE_TASKS so it actually extracts the evaluated scores!
ALL_TASKS = REASONING_TASKS + CODE_TASKS

BATCH_SIZE = "auto"  # Use "auto" to let LM-Eval determine optimal batch size based on GPU memory


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def extract_acc(metrics):
    """Safely extract accuracy or pass@1 from lm-eval output dict with fallback search."""
    # 1. Check preferred exact keys first
    preferred_keys = ["pass@1,none", "pass@1", "acc,none", "acc_norm,none", "acc"]
    for key in preferred_keys:
        if key in metrics:
            return metrics[key]

    # 2. Fallback: search for any key containing 'pass' or 'acc'
    for key, value in metrics.items():
        if "pass" in key.lower() or "acc" in key.lower():
            return value

    # 3. If still nothing found, log keys for debugging
    logging.error(f"Could not find accuracy/pass@1 metric in keys: {list(metrics.keys())}")
    return 0.0


def run_evaluations(lm_eval_model):
    """Safely runs reasoning (5-shot) and code tasks (0-shot) separately and merges results."""
    results_dict = {}

    if REASONING_TASKS:
        reasoning_eval = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=REASONING_TASKS,
            num_fewshot=5,
            batch_size=BATCH_SIZE,
        )
        results_dict.update(reasoning_eval["results"])

    if CODE_TASKS:
        code_eval = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=CODE_TASKS,
            num_fewshot=0,
            batch_size=BATCH_SIZE,
            # Add this exact flag below to allow Python execution
            confirm_run_unsafe_code=True,
        )
        results_dict.update(code_eval["results"])

    return results_dict


def save_results_df(results_list, csv_out_path):
    """Formats columns cleanly (Delta, Step_Used, Individual Tasks..., Average_Accuracy) and saves CSV."""
    df = pd.DataFrame(results_list)

    # Dynamically order columns: Delta & Step_Used first, individual tasks middle, Average_Accuracy last
    base_cols = ["Delta", "Step_Used"]
    end_cols = ["Average_Accuracy"]
    task_cols = [c for c in df.columns if c not in base_cols and c not in end_cols]

    ordered_cols = base_cols + task_cols + end_cols
    df = df[ordered_cols]

    # Round all numeric columns to 1 decimal place and save
    df.to_csv(csv_out_path, index=False, float_format="%.3f")

    return df, task_cols


def main():
    # 1. Parse Arguments & Dynamic Configurations
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    MODEL_ID = arguments.model_id

    model_config = model_configurations.MODELS_CONFIG.get(MODEL_ID)
    model_name = model_config.model_name

    eval_out_dir = os.path.join(root_folder, "outputs", "4_evaluate_utility", model_name)
    os.makedirs(eval_out_dir, exist_ok=True)

    # Optional: changed file name to distinguish it's just the baseline
    csv_out_path = os.path.join(eval_out_dir, f"utility_results_baseline_{model_name}.csv")

    logging.info(f"=== STARTING DOWNSTREAM UTILITY VERIFICATION (BASELINE ONLY) ({model_name}) ===")

    # 2. Load Base Model
    logging.info(f"Loading Base Model ({MODEL_ID})...")
    base_model, tokenizer = utils.load_model(MODEL_ID)
    base_model.eval()

    results_list = []

    # ==========================================
    # PHASE 1: EVALUATE CLEAN BASELINE
    # ==========================================
    logging.info("--- EVALUATING CLEAN BASELINE ---")

    # Wrap base model for LM-Eval Harness
    lm_eval_model = HFLM(pretrained=base_model, tokenizer=tokenizer, batch_size=BATCH_SIZE, chat_template=True)

    baseline_results = run_evaluations(lm_eval_model)

    baseline_row = {"Delta": "0.00", "Step_Used": "Clean"}
    total_acc = 0.0
    num_tasks = 0

    # Iterate over ALL_TASKS explicitly to ignore subtasks
    for task_name in ALL_TASKS:
        if task_name in baseline_results:
            acc = extract_acc(baseline_results[task_name])
            baseline_row[task_name] = acc
            total_acc += acc
            num_tasks += 1
        else:
            logging.warning(f"Task {task_name} was not found in evaluation results! Returning 0.0.")
            baseline_row[task_name] = 0.0
            num_tasks += 1

    baseline_row["Average_Accuracy"] = total_acc / num_tasks if num_tasks > 0 else 0.0
    results_list.append(baseline_row)

    # Save baseline immediately
    df, task_cols = save_results_df(results_list, csv_out_path)
    logging.info(f"Baseline Average Accuracy: {baseline_row['Average_Accuracy']:.2%}")

    # ==========================================
    # PHASE 2: VERDICT
    # ==========================================
    logging.info("=" * 85)
    logging.info(f"{'Delta':<8} | {'Step Used':<20} | " + " | ".join([f"{t[:10]:<10}" for t in task_cols]) + f" | {'Average':<10}")
    logging.info("-" * 85)

    for r in results_list:
        task_str = " | ".join([f"{r.get(t, 0.0):<10.2%}" for t in task_cols])
        logging.info(f"{r['Delta']:<8} | {r['Step_Used']:<20} | {task_str} | {r['Average_Accuracy']:<10.2%}")

    logging.info("=" * 85)
    logging.info(f"Baseline evaluation complete. CSV output saved to: {csv_out_path}")


if __name__ == "__main__":
    main()
