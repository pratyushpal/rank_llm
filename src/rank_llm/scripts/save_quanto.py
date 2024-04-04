"""Converts and stores Quantized using quanto."""

import argparse
import json
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize, safe_save, safe_load
import torch 

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="castorini/rank_zephyr_7b_v1_full",
        help="Path/slug to the original model.",
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default="quanto_rank_zephyr_7b_v1_full",
        help="Path/slug where the quantized model is to be stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to use for generation."
    )
    parser.add_argument(
        "--weights",
        type=str, 
        default="int8", 
        choices=["int4", "int8", "float8"]
    )
    parser.add_argument(
        "--activations",
        type=str, 
        default="int8",
        choices=["none", "int8", "float8"]
    )
    
    args = parser.parse_args()

    return args

def keyword_to_itype(k):
    return {
        "none": None,
        "int4": qint4,
        "int8": qint8,
        "float8": qfloat8,
    }[k]


def main():
    """Entry point of the script."""
    args = parse_args()
    model_path = args.model_path
    quant_path = args.quant_path

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)


    # Load model
    logging.info(f"Loading model from {model_path}.")

    model =  AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    weights = keyword_to_itype(args.weights)
    # As serialization is only supported for weights 
    #activations = keyword_to_itype(args.activations)

    #Only accepting activations as None for now
    logging.info(f"Quantizing the model.")
    quantize(model, weights=weights)
    
    logging.info(f"Freezing model weights.")
    freeze(model)
    logging.info(f"Saving quantized model at {quant_path}.")
    safe_save(model.state_dict(), quant_path)
    tokenizer.save_pretrained(quant_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()