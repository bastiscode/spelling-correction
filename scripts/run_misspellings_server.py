import argparse
import os
import logging
import time
import threading
from typing import List, Dict, Any

import torch
import transformers
from flask import Flask, abort, request, jsonify, Response

from text_correction_utils.logging import get_logger

_LANG_TO_NAME = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["alpaca-7b", "google/flan-ul2", "bigscience/bloomz-560m"],
        default="alpaca-7b",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--words-per-prompt", type=int, default=10)
    parser.add_argument("--port", type=int, default=40000)
    return parser.parse_args()


def run_server(args: argparse.Namespace):
    assert torch.cuda.is_available(), "CUDA GPU is required"
    if args.model == "alpaca-7b":
        args.model = "chavinlo/alpaca-native"

    if args.model == "chavinlo/alpaca-native":
        model = transformers.LlamaForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "decapoda-research/llama-7b-hf"     
        )
        args.batch_size = 1
    elif args.model == "google/flan-ul2":
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    model = model.eval()
    torch.compile(model)

    lock = threading.Lock()
    logger = get_logger("SERVER")
    app = Flask("Misspellings generator")
    logging.getLogger("werkzeug").disabled = True
    os.environ["FLASK_DEBUG"] = "development"
    # app.logger.disabled = True

    @app.after_request
    def after_request(response: Response) -> Response:
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    @app.route("/generate", methods=["POST"])
    def generate() -> Response:
        data: Dict[str, Any] = request.get_json()
        if ("lang" not in data
            or data["lang"] not in _LANG_TO_NAME
                or "words" not in data):
            return abort(400)
        prompts = []
        num = int(data.get("misspellings", 10))
        start = time.perf_counter()
        for i in range(0, len(data["words"]), args.words_per_prompt):
            prompt = get_prompt(
                data["lang"],
                num,
                data["words"][i:i + args.words_per_prompt],
                is_alpaca=args.model == "chavinlo/alpaca-native"
            )
            prompts.append(prompt)
        prompt_str = "\n\n".join(prompts)
        logger.info(
            f"running the following prompts:\n{prompt_str}"
        )
        with lock, torch.inference_mode():
            decoded = []
            for i in range(0, len(prompts), args.batch_size):
                prompt_batch = prompts[i:i + args.batch_size]
                inputs = tokenizer(
                    prompt_batch if len(prompt_batch) > 1 else prompt_batch[0], 
                    return_tensors="pt", 
                    padding=len(prompt_batch) > 1
                ).to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=2048)
                decoded.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        decoded_str = "\n\n".join(decoded)
        logger.info(
            f"generated {num} misspellings for {len(data['words'])} words in {time.perf_counter() - start:.2f} seconds:\n{decoded_str}"
        )
        return jsonify({})

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


def get_prompt(
    lang: str,
    num: int,
    words: List[str],
    is_alpaca: bool
) -> str:
    prompt = f"""For each of the following {lang} words, give {num} common and unique misspellings of it.
Use the following format for each word: <word>; <misspelling1>, <misspelling2>, ..., <misspelling{num}>
Mark the beginning of your answer with: ### START ###

Words:
"""
    prompt += "\n".join(f"- {word}" for word in words)

    if is_alpaca:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

    return prompt


if __name__ == "__main__":
    run_server(parse_args())
