# WoE Wrote It? Watermarking Mixture-of-Experts LLMs for Black-Box Text Provenance

---
## Overview

The framework is organized into five sequential pipeline steps for profiling, watermarking, and evaluating Mixture-of-Experts (MoE) models.

Step 1: [1_profile_and_setup.py](1_profile_and_setup.py) profiles router activation counts across MoE layers using a calibration dataset.

Step 2: [2_generate_payload.py](2_generate_payload.py) identifies target expert subsets across layer quartiles and generates the cryptographic token greenlist payload.

Step 3: [3_inject_watermark.py](3_inject_watermark.py) injects the watermark by fine-tuning targeted expert feed-forward matrices with LoRA and a custom routing loss.

Step 4: [4_evaluate_utility.py](4_evaluate_utility.py) measures downstream benchmark utility and task performance using the evaluation harness.

Step 5: [5_evaluate_watermark.py](5_evaluate_watermark.py) runs watermark calibration and ablation sweeps to evaluate generation perplexity, TPR, and routing provenance $Z$-scores.