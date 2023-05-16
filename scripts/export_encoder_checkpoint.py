import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def export(args: argparse.Namespace):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}

    for prefix, replacement in [
        ("encoder.", "encoder."),
        ("encoder_embedding.", "embedding.")
    ]:
        for key in state_dict:
            if key.startswith(prefix):
                new_key = replacement + key[len(prefix):]
                new_state_dict[new_key] = state_dict[key]

    checkpoint["model_state_dict"] = new_state_dict
    torch.save(checkpoint, args.out)


if __name__ == "__main__":
    export(parse_args())
