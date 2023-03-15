import copy
from typing import Dict, Any, Optional, Tuple, Union
from text_correction_utils.modules.grouping import Grouping

import torch
from torch import nn

from text_correction_utils import tokenization
from text_correction_utils.modules.embedding import (
    Embedding,
    StandardEmbedding,
    embedding_from_config
)
from text_correction_utils.modules.encoder import Encoder, encoder_from_config
from text_correction_utils.modules.decoder import Decoder, TransformerDecoder, decoder_from_config
from text_correction_utils.modules.head import (
    Head,
    SequenceClassificationHead,
    head_from_config
)


class WordCorrectionHead(Head):
    def __init__(
        self,
        embedding: StandardEmbedding,
        dim: int,
        num_layers: int,
        heads: int,
        ffw_dim: int,
        dropout: float,
        with_pos: Optional[str] = None,
        activation: str = "gelu",
        word_group_lengths: str = "word_group_lengths"
    ):
        self.embedding = embedding
        assert isinstance(self.embedding, StandardEmbedding)

        self.decoder = TransformerDecoder(
            dim,
            num_layers,
            heads,
            ffw_dim,
            dropout,
            with_pos,
            memories=["characters"],
            activation=activation
        )

        emb_shape = self.embedding.embedding.emb.weight.shape
        self.head = nn.Linear(emb_shape[1], emb_shape[0])
        self.head.weight = self.embedding.embedding.emb.weight

        self.word_group_lengths = word_group_lengths

    def forward(
        self,
        x: torch.Tensor,
        word_token_ids: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert word_token_ids is not None
        memory_attn_mask = ...
        dec, pos_emb = self.embedding(word_token_ids, **kwargs)
        dec = self.decoder(
            dec,
            pos_emb,
            memories={"characters": x},
            memory_attn_masks={"characters": memory_attn_mask},
            **kwargs
        )
        return self.head(dec)


class WordDetectionHead(Head):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        dropout: float,
        activation: str = "gelu",
        word_group_name: str = "word_groups",
        word_group_lengths: str = "word_group_lengths",
        word_group_padding_mask: str = "word_group_padding_mask",
        word_features: Optional[str] = None,
    ):
        super().__init__()
        self.word_features = word_features
        self.word_grouping = Grouping(
            group_name=word_group_name,
            group_lengths=word_group_lengths,
            group_padding_mask=word_group_padding_mask
        )
        self.head = SequenceClassificationHead(
            dim,
            2,
            num_layers,
            dropout,
            activation
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        word_x, kwargs = self.word_grouping(x, **kwargs)
        if self.word_features is not None:
            assert self.word_features in kwargs, \
                f"expected {self.word_features} in kwargs"
            word_x = torch.cat([word_x, kwargs[self.word_features]], dim=-1)
        return self.head(word_x, **kwargs)


class Model(nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class SpellingCorrectionModel(Model):
    def __init__(
        self,
        embedding: Embedding,
        encoder: Encoder,
        whitespace_head: SequenceClassificationHead,
        detection_head: WordDetectionHead,
        correction_head: WordCorrectionHead,
        word_encoder: Optional[Encoder] = None,
        whitespace_teacher_forcing_ratio: float = 0.5
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder

        self.whitespace_head = whitespace_head
        self.detection_head = detection_head
        self.correction_head = correction_head

        self.tfr = whitespace_teacher_forcing_ratio

        if word_encoder is not None:
            self.word_encoder = word_encoder

    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        emb, pos_emb = self.embedding(token_ids, **kwargs)
        enc, kwargs = self.encoder(emb, pos=pos_emb, **kwargs)
        outputs = {}
        outputs["wsc"] = self.whitespace_head(enc, **kwargs)
        outputs["sedw"] = self.detection_head(enc, **kwargs)
        outputs["sec"] = self.correction_head(enc, **kwargs)
        return outputs, self.encoder.additional_losses()


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
        self.decoder = decoder

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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        emb, pos_emb = self.encoder_embedding(token_ids, **kwargs)
        enc, kwargs = self.encoder(emb, pos_emb, **kwargs)
        return enc, kwargs

    def decode(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        enc: torch.Tensor,
        enc_padding_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        emb, pos_emb = self.decoder_embedding(token_ids, **kwargs)
        dec, kwargs = self.decoder(
            emb,
            pos_emb,
            padding_mask=padding_mask,
            memories={self.memory_name: enc},
            memory_padding_masks={self.memory_name: enc_padding_mask},
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
        enc_padding_mask = kwargs.pop(self.memory_padding_mask)
        dec = self.decode(
            target_token_ids,
            target_padding_mask,
            enc,
            enc_padding_mask,
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
