"""Converts and stores Quantized using quanto."""

import argparse
import json
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize, safe_save, safe_load
import torch 
import time


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

@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    start = time.time()
    outputs = model.generate(
        input_ids=inputs.input_ids.to(device),
        max_new_tokens=max_new_tokens,
        attention_mask=inputs.attention_mask.to(device),
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )
    end = time.time()
    generated_text = tokenizer.decode(outputs[0])
    print(f"Generated '{generated_text}' in [{end - start:.2f} s]")


@torch.no_grad()
def calibrate(model, tokenizer, dataset, device, batch_size, samples=None):
    model.eval()
    total = 0
    for batch in dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        model(input_ids, attention_mask=attention_mask)
        total += input_ids.size(0)
        if samples is not None and total >= samples:
            break

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

    model =  AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    weights = keyword_to_itype(args.weights)
    logging.info(f"State dict of the base model.")
    print(model)
    print(model.state_dict().keys())

    logging.info(f"Model Keys.")
    print(len(model.state_dict().keys()))


    # As serialization is only supported for weights 
    #activations = keyword_to_itype(args.activations)

    #Only accepting activations as None for now
    logging.info(f"Quantizing the model.")
    quantize(model, weights=weights)
    
    logging.info(f"Freezing model weights.")
    freeze(model)

    print(model)

    logging.info(f"State dict of the quantized model.")
    print(model.state_dict().keys())

    logging.info(f"Quantized Model Keys.")
    print(len(model.state_dict().keys()))
    logging.info(f"Saving quantized model at {quant_path}.")
    tokenizer.save_pretrained(quant_path)
    safe_save(model.state_dict(), quant_path+"/"+quant_path+".pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()