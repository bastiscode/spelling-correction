import copy
from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn

from text_correction_utils import tokenization
from text_correction_utils.modules.embedding import (
    Embedding,
    StandardEmbedding,
    embedding_from_config
)
from text_correction_utils.modules.encoder import (
    Encoder,
    encoder_from_config
)
from text_correction_utils.modules.decoder import (
    Decoder,
    decoder_from_config
)
from text_correction_utils.modules.head import (
    Head,
    head_from_config
)


class Model(nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class EncoderDecoderWithHead(Model):
    def __init__(
        self,
        encoder_embedding: Embedding,
        encoder: Encoder,
        decoder: Decoder,
        memory_name: str = "encoder",
        memory_padding_mask: str = "padding_mask",
        head: Optional[Head] = None,
        decoder_embedding: Optional[Embedding] = None,
    ):
        super().__init__()
        self.encoder_embedding = encoder_embedding
        self.encoder = encoder

        assert isinstance(self.encoder_embedding, StandardEmbedding), \
            "encoder embedding must be a standard embedding"
        self.share_encoder_decoder_embedding = decoder_embedding is None
        self.share_decoder_embedding_with_head = head is None
        if self.share_encoder_decoder_embedding:
            self.decoder_embedding = self.encoder_embedding
        else:
            assert decoder_embedding is not None, \
                "decoder_embedding must be provided if not sharing encoder embedding with decoder embedding"
            self.decoder_embedding = decoder_embedding

        self.decoder = decoder

        assert isinstance(self.decoder_embedding, StandardEmbedding), \
            "decoder embedding must be a standard embedding"
        if self.share_decoder_embedding_with_head:
            emb_shape = self.decoder_embedding.embedding.emb.weight.shape
            self.head = nn.Linear(emb_shape[1], emb_shape[0])
            self.head.weight = self.decoder_embedding.embedding.emb.weight
        else:
            assert head is not None and isinstance(head, Head), \
                "head must be provided if not sharing decoder embedding with head"
            self.head = head

        self.memory_name = memory_name
        self.memory_padding_mask = memory_padding_mask

    def encode(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        emb, pos_emb = self.encoder_embedding(token_ids, **kwargs)
        enc, kwargs = self.encoder(emb, pos_emb, **kwargs)
        kwargs["memory_padding_masks"] = {
            self.memory_name: kwargs.pop(self.memory_padding_mask)
        }
        return {self.memory_name: enc}, kwargs

    def decode(
        self,
        token_ids: torch.Tensor,
        enc: Dict[str, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.training:
            assert padding_mask is not None, \
                "padding_mask must be provided during training"
            kwargs["padding_mask"] = padding_mask
        emb, pos_emb = self.decoder_embedding(token_ids, **kwargs)
        dec, kwargs = self.decoder(
            emb,
            pos_emb,
            memories=enc,
            **kwargs
        )
        if self.share_decoder_embedding_with_head:
            return self.head(dec)
        else:
            return self.head(dec, **kwargs)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_token_ids: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert target_token_ids is not None and target_padding_mask is not None
        enc, kwargs = self.encode(token_ids, **kwargs)
        dec = self.decode(
            target_token_ids,
            enc,
            padding_mask=target_padding_mask,
            **kwargs
        )
        additional_losses = self.encoder.additional_losses()
        for k, v in self.decoder.additional_losses().items():
            if k in additional_losses:
                additional_losses[k] += v
            else:
                additional_losses[k] = v
        return dec, {
            **self.encoder.additional_losses(),
            **self.decoder.additional_losses()
        }


def model_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: Optional[tokenization.Tokenizer],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")

    if model_type == "encoder_with_heads":
        raise NotImplementedError
        # embedding = embedding_from_config(cfg["embedding"], input_tokenizer)
        # encoder = encoder_from_config(cfg["encoder"])
        # head = head_from_config(cfg["head"])
        # return SpellingCorrectionModel(embedding, encoder, head)

    elif model_type == "encoder_decoder_with_head":
        encoder_embedding = embedding_from_config(
            cfg["encoder_embedding"], input_tokenizer
        )
        encoder = encoder_from_config(cfg["encoder"])
        decoder = decoder_from_config(cfg["decoder"])
        # if there is no decoder embedding, this means we share the
        # encoder embedding with the decoder embedding
        if "decoder_embedding" in cfg:
            decoder_embedding = embedding_from_config(
                cfg["decoder_embedding"], output_tokenizer
            )
        else:
            decoder_embedding = None
        # if there is no head, this means we share the decoder
        # embedding with the head
        if "head" in cfg:
            head = head_from_config(cfg["head"])
        else:
            head = None
        return EncoderDecoderWithHead(
            encoder_embedding,
            encoder,
            decoder,
            memory_name=cfg.get("memory", "encoder"),
            head=head,
            decoder_embedding=decoder_embedding
        )

    else:
        raise ValueError(f"unknown model type {model_type}")
