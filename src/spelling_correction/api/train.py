import os
from typing import Dict, Any, Tuple

import torch
from torch import nn

from text_correction_utils.api.trainer import Trainer
from text_correction_utils import tokenization, data, api

from spelling_correction.model import model_from_config


class SpellingCorrectionTrainer(Trainer):
    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
        input_tokenizer = tokenization.Tokenizer.from_config(
            cfg["input_tokenizer"]
        )
        if "output_tokenizer" in cfg:
            output_tokenizer = tokenization.Tokenizer.from_config(
                cfg["output_tokenizer"]
            )
        else:
            output_tokenizer = None
        return model_from_config(
            cfg["model"],
            input_tokenizer,
            output_tokenizer
        )

    def _prepare_batch(
        self,
        batch: data.DataBatch
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        assert len(batch) > 0, "got empty batch"
        max_length = self.cfg["train"]["data"]["max_length"]

        (
            token_ids_np,
            pad_mask_np,
            lengths,
            info,
            labels_np,
            label_info
        ) = batch.tensors()

        token_ids_np = token_ids_np[:, :max_length]
        pad_mask_np = pad_mask_np[:, :max_length]
        lengths = [min(length, max_length) for length in lengths]

        inputs = {
            "token_ids": torch.from_numpy(token_ids_np).to(
                non_blocking=True,
                device=self.info.device
            ),
            "lengths": lengths,
            "padding_mask": torch.from_numpy(pad_mask_np).to(
                non_blocking=True,
                device=self.info.device
            ),
            **api.to(info, self.info.device)
        }

        labels_np = labels_np[:, :max_length]
        labels = torch.from_numpy(labels_np).to(
            non_blocking=True,
            dtype=torch.long,
            device=self.info.device
        )

        if self.cfg["model"]["type"] == "encoder_decoder_with_head":
            # for encoder decoder models we need to provide additional
            # information for the targets
            label_info = api.to(label_info, self.info.device)

            inputs["target_token_ids"] = label_info.pop(
                "token_ids"
            )[:, :max_length]
            inputs["target_padding_mask"] = label_info.pop(
                "padding_mask"
            )[:, :max_length]
            inputs["target_lengths"] = [
                min(length, max_length)
                for length in label_info.pop("lengths")
            ]

            for k, v in label_info.items():
                inputs[f"target_{k}"] = v

        return inputs, labels


def main():
    parser = SpellingCorrectionTrainer.parser(
        "Train spelling correction", "Train a spelling correction model"
    )
    args = parser.parse_args()
    work_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
    if args.platform == "local":
        SpellingCorrectionTrainer.train_local(
            work_dir, args.experiment, args.config, args.profile
        )
    else:
        SpellingCorrectionTrainer.train_slurm(
            work_dir, args.experiment, args.config
        )


if __name__ == "__main__":
    main()
