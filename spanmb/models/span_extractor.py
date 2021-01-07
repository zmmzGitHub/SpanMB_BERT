import torch
from torch.nn.parameter import Parameter
from overrides import overrides

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import util

from allennlp.common.checks import ConfigurationError

class MaxSpanExtractor(SpanExtractor):

    def __init__(
            self,
            input_dim: int,
            num_width_embeddings: int = None,
            span_width_embedding_dim: int = None,
            bucket_widths: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings, embedding_dim=span_width_embedding_dim
            )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_buckets and span_width_embedding_dim."
            )
        else:
            self._span_width_embedding = None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        if self._span_width_embedding is not None:
            return self._input_dim + self._span_width_embedding.get_output_dim()
        return self._input_dim

    @overrides
    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor = None,
            span_indices_mask: torch.BoolTensor = None,
    ) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask
        # [batch_size, span_num, max_span_width, emb_size]
        span_embeddings, span_mask = util.batched_span_select(sequence_tensor, span_indices)
        # set unmask embeddings to 0
        span_embeddings = span_embeddings * span_mask.unsqueeze(-1).float()
        # [batch_size, span_num, emb_size]
        span_max_embeddings = torch.max(span_embeddings, 2)[0]  # span_mask

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = util.bucket_values(
                    span_ends - span_starts, num_total_buckets=self._num_width_embeddings
                )
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            span_max_embeddings = torch.cat([span_max_embeddings, span_width_embeddings], -1)

        if span_indices_mask is not None:
            return span_max_embeddings * span_indices_mask.unsqueeze(-1)

        return span_max_embeddings
