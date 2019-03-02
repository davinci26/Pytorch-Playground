import logging
import json

def load_parameters(filepath):
    default_training_parameters = {
                                    "epoch": 2,
                                    "model_name": "LeNet",
                                    "learning_rate": 0.001,
                                    "momentum": 0.9,
                                    "checkpoint_interval" : 2000,
                                    "calculate_validation": True,
                                    "hyperparameter_optimization":
                                    {
                                        "optimize": True,
                                        "steps": 3
                                    },
                                    "description": "Baseline LeNet model - Pytorch example"
                                }
    if filepath is None or filepath.isspace():
        logging.info("No training parameteres specified... Using default values")
        training_parameters = default_training_parameters
    else:
        with open(filepath) as f:
            training_parameters = json.load(f)
    return training_parameters