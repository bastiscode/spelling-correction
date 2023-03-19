import argparse
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
        choices=["google/flan-ul2", "bigscience/bloomz-560m"],
        required=True
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--port", type=int, default=40000)
    return parser.parse_args()


def run_server(args: argparse.Namespace):
    assert torch.cuda.is_available(), "CUDA GPU is required"
    if args.model == "google/flan-ul2":
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto"
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    lock = threading.Lock()
    logger = get_logger("SERVER")
    app = Flask("Misspellings generator")

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
        for i in range(0, len(data["words"]), args.batch_size):
            prompt = get_prompt(
                data["lang"],
                num,
                data["words"][i:i + args.batch_size]
            )
            prompts.append(prompt)
        with lock:
            inputs = tokenizer(prompts, return_tensors="pt")
            outputs = model.generate(**inputs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info(
            f"generated {num} misspellings for {len(data['words'])} words in {time.perf_counter() - start:.2f} seconds")
        return jsonify({})

    app.run(host="0.0.0.0", port=args.port)


def get_prompt(
    lang: str,
    num: int,
    words: List[str]
) -> str:
    prompt = f"""
For each of the following {lang} words, give {num} common and unique misspellings of it.
Use the following format for each word: <word>; <misspelling1>, <misspelling2>, ..., <misspelling{num}>
Mark the beginning of your answer with: ### START ###

Words:
"""
    for word in words:
        prompt += f"- {word}\n"
    return prompt


if __name__ == "__main__":
    run_server(parse_args())
