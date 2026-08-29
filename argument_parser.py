import argparse
import model_configurations

# Dynamically extract the master list of MoE models
MODELS = list(model_configurations.MODELS_CONFIG.keys())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for WoE profiling.")

    parser.add_argument("--root", default="", type=str)
    parser.add_argument("--delta", type=float, default=1.0, help="Watermark strength penalty")
    parser.add_argument("--model_idx", type=int, default=0, help="Integer index mapping to the MoE model list")

    arguments = parser.parse_args()

    # Assign the model ID based on the dynamic list
    arguments.model_id = MODELS[arguments.model_idx]

    return arguments
