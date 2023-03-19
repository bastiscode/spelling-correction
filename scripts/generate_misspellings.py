import argparse
import os
import requests

from tqdm import tqdm

from text_correction_utils.logging import get_logger

_LANG_TO_NAME = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True,
                        choices=list(_LANG_TO_NAME))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-misspellings", type=int, default=10)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=40000)
    return parser.parse_args()


def generate(args: argparse.Namespace):
    logger = get_logger("MISSPELLINGS")
    address = f"{args.host}:{args.port}"

    with open(args.dictionary, "r", encoding="utf8") as inf:
        words = [line.strip().split("\t")[0] for line in inf]

    logger.info(f"generating misspellings for {len(words)} words")

    exists = os.path.exists(args.output)
    if exists:
        with open(args.output, "r", encoding="utf8") as inf:
            start_at = len(inf.readlines())
        words = words[start_at:]
        logger.info(f"skipping the first {start_at} words")
    else:
        dirname = os.path.dirname(args.output)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

    with open(args.output, "a", encoding="utf8") as of:
        for i in range(0, len(words), args.batch_size):
            word_batch = words[i:i + args.batch_size]

            misspellings = None
            try:
                response = requests.post(
                    f"http://{address}/generate",
                    json={
                        "words": word_batch,
                        "lang": args.lang,
                        "misspellings": args.num_misspellings
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if "misspellings" in data:
                        misspellings = data["misspellings"]
            except Exception as e:
                logger.info(f"error getting misspellings: {e}")

            if misspellings is None:
                logger.error(
                    f"could not generate misspellings for words: {word_batch}")
                for word in word_batch:
                    of.write(f"### ERROR ### {word}\n")
            exit()


if __name__ == "__main__":
    generate(parse_args())
