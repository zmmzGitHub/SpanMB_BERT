import logging
from typing import Dict, List, Optional, Union
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, TimeDistributed
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

# Import submodules.
from spanmb.models.ner import NERTagger
from spanmb.models.relation import RelationExtractor
from spanmb.data.dataset_readers import document
from spanmb.models.span_extractor import MaxSpanExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("spanmb")
class SpanMB(Model):
    """
    TODO(dwadden) document me.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    submodule_params: ``TODO(dwadden)``
        A nested dictionary specifying parameters to be passed on to initialize submodules.
    max_span_width: ``int``
        The maximum width of candidate spans.
    target_task: ``str``:
        The task used to make early stopping decisions.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    module_initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the individual modules.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    display_metrics: ``List[str]``. A list of the metrics that should be printed out during model
        training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 modules,  # TODO(dwadden) Add type.
                 feature_size: int,
                 max_span_width: int,
                 target_task: str,
                 feedforward_params: Dict[str, Union[int, float]],
                 loss_weights: Dict[str, float],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 module_initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None) -> None:
        super(SpanMB, self).__init__(vocab, regularizer)

        ####################

        # Create span extractor.
        self._endpoint_span_extractor = EndpointSpanExtractor(
            embedder.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False)
        self._max_span_extractor = MaxSpanExtractor(
            embedder.get_output_dim()
        )
        # max pooling on span and span width feature
        # self._max_span_extractor = MaxSpanExtractor(
        #     embedder.get_output_dim(),
        #     num_width_embeddings=max_span_width,
        #     span_width_embedding_dim=feature_size,
        #     bucket_widths=False
        # )
        ####################

        # Set parameters.
        self._embedder = embedder
        self._loss_weights = loss_weights
        self._max_span_width = max_span_width
        self._display_metrics = self._get_display_metrics(target_task)
        token_emb_dim = self._embedder.get_output_dim()
        span_emb_dim = self._endpoint_span_extractor.get_output_dim() + self._max_span_extractor.get_output_dim()

        ####################

        # Create submodules.

        modules = Params(modules)

        # Helper function to create feedforward networks.
        def make_feedforward(input_dim):
            return FeedForward(input_dim=input_dim,
                               num_layers=feedforward_params["num_layers"],
                               hidden_dims=feedforward_params["hidden_dims"],
                               activations=torch.nn.ReLU(),
                               dropout=feedforward_params["dropout"])

        # Submodules

        self._ner = NERTagger.from_params(vocab=vocab,
                                          make_feedforward=make_feedforward,
                                          span_emb_dim=span_emb_dim,
                                          params=modules.pop("ner"))

        self._relation = RelationExtractor.from_params(vocab=vocab,
                                                       make_feedforward=make_feedforward,
                                                       token_emb_dim=token_emb_dim,
                                                       span_emb_dim=span_emb_dim,
                                                       feature_size=feature_size,
                                                       params=modules.pop("relation"))

        ####################

        # Initialize text embedder and all submodules
        for module in [self._ner, self._relation]:
            module_initializer(module)

        initializer(self)

    @staticmethod
    def _get_display_metrics(target_task):
        """
        The `target` is the name of the task used to make early stopping decisions. Show metrics
        related to this task.
        """
        lookup = {
            "ner": [f"MEAN__{name}" for name in
                    ["ner_precision", "ner_recall", "ner_f1"]],
            "relation": [f"MEAN__{name}" for name in
                         ["relation_precision", "relation_recall", "relation_f1"]]}
        if target_task not in lookup:
            raise ValueError(f"Invalied value {target_task} has been given as the target task.")
        return lookup[target_task]

    @staticmethod
    def _debatch(x):
        # TODO(dwadden) Get rid of this when I find a better way to do it.
        return x if x is None else x.squeeze(0)

    @overrides
    def forward(self,
                text,
                spans,
                metadata,
                ner_labels=None,
                ner_nested=None,
                relation_labels=None):
        """
        TODO(dwadden) change this.
        """
        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        if relation_labels is not None:
            relation_labels = relation_labels.long()

        # TODO(dwadden) Multi-document minibatching isn't supported yet. For now, get rid of the
        # extra dimension in the input tensors. Will return to this once the model runs.
        if len(metadata) > 1:
            raise NotImplementedError("Multi-document minibatching not yet supported.")

        metadata = metadata[0]
        spans = self._debatch(spans)  # (n_sents, max_n_spans, 2)
        ner_labels = self._debatch(ner_labels)  # (n_sents, max_n_spans)
        if ner_nested is not None:
            ner_nested = self._debatch(ner_nested)
        relation_labels = self._debatch(relation_labels)  # (n_sents, max_n_spans, max_n_spans)
        # print("batch_si=ze: ", spans.size(0))
        # Encode using BERT, then debatch.
        # Since the data are batched, we use `num_wrapping_dims=1` to unwrap the document dimension.
        # (1, n_sents, max_sententence_length, embedding_dim)

        # TODO(dwadden) Deal with the case where the input is longer than 512.
        text_embeddings = self._embedder(text, num_wrapping_dims=1)
        # (n_sents, max_n_wordpieces, embedding_dim)
        text_embeddings = self._debatch(text_embeddings)

        # (n_sents, max_sentence_length)
        text_mask = self._debatch(util.get_text_field_mask(text, num_wrapping_dims=1).float())
        sentence_lengths = text_mask.sum(dim=1).long()  # (n_sents)

        span_mask = (spans[:, :, 0] >= 0).float()  # (n_sents, max_n_spans)
        # SpanFields return -1 when they are used as padding. As we do some comparisons based on
        # span widths when we attend over the span representations that we generate from these
        # indices, we need them to be <= 0. This is only relevant in edge cases where the number of
        # spans we consider after the pruning stage is >= the total number of spans, because in this
        # case, it is possible we might consider a masked span.
        spans = F.relu(spans.float()).long()  # (n_sents, max_n_spans, 2)

        # Shape: (batch_size, num_spans, 3 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
        max_span_embeddings = self._max_span_extractor(text_embeddings, spans)
        span_embeddings = torch.cat([max_span_embeddings, span_embeddings], -1)

        # Make calls out to the modules to get results.
        output_ner = {'loss': 0}
        output_relation = {'loss': 0}
        # # Prune and compute span representations for relation module
        # if self._loss_weights["relation"] > 0 or self._relation.rel_prop > 0:
        #     output_relation = self._relation.compute_representations(
        #         spans, span_mask, span_embeddings, text_embeddings, sentence_lengths, metadata)

        # if self._relation.rel_prop > 0:
        #     output_relation = self._relation.relation_propagation(output_relation)
        #     span_embeddings = self.update_span_embeddings(span_embeddings, span_mask,
        #                                                 output_relation["top_span_embeddings"],
        #                                                 output_relation["top_span_mask"],
        #                                                 output_relation["top_span_indices"])

        # Prune and compute span representations for relation module
        if self._loss_weights["relation"] > 0:
            output_relation = self._relation.compute_representations(
                spans, span_mask, span_embeddings, text_embeddings, sentence_lengths, metadata)

        # Make predictions and compute losses for each module
        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, ner_nested, metadata)
            top_ner_predicted_labels, top_ner_gold_labels = self.get_top_span_ner_labels(
                output_ner["predicted_ner"], output_ner["span_mask"], ner_labels,
                output_relation["top_span_mask"], output_relation["top_span_indices"])
            if self._loss_weights["relation"] > 0:
                output_relation["top_ner_predicted_labels"] = top_ner_predicted_labels
                output_relation["top_ner_gold_labels"] = top_ner_gold_labels

        if self._loss_weights['relation'] > 0:
            output_relation = self._relation.predict_labels(relation_labels, output_relation, metadata)

        # if self._loss_weights['relation'] > 0:
        #     output_relation = self._relation(
        #         spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)

        # Use `get` since there are some cases where the output dict won't have a loss - for
        # instance, when doing prediction.
        loss = (self._loss_weights['ner'] * output_ner.get("loss", 0) +
                self._loss_weights['relation'] * output_relation.get("loss", 0))

        # Multiply the loss by the weight multiplier for this document.
        weight = metadata.weight if metadata.weight is not None else 1.0
        loss *= torch.tensor(weight)

        output_dict = dict(relation=output_relation,
                           ner=output_ner)
        output_dict['loss'] = loss

        output_dict["metadata"] = metadata
        # torch.cuda.empty_cache()
        return output_dict

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings,
                               top_span_mask, top_span_indices):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr,
                                    span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

    def get_top_span_ner_labels(self, predicted_ner, span_mask, gold_ner_labels, top_span_mask, top_span_indices):
        top_predicted_ner = top_span_mask.new_zeros([top_span_mask.size(0), top_span_mask.size(1)])
        top_gold_ner = top_span_mask.new_zeros([top_span_mask.size(0), top_span_mask.size(1)])
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                top_predicted_ner[sample_nr, top_span_nr] = predicted_ner[sample_nr, span_nr]
                if gold_ner_labels is not None:
                    top_gold_ner[sample_nr, top_span_nr] = gold_ner_labels[sample_nr, span_nr]

        return top_predicted_ner, top_gold_ner

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:

        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        doc = copy.deepcopy(output_dict["metadata"])

        if self._loss_weights["ner"] > 0:
            for predictions, sentence in zip(output_dict["ner"]["predictions"], doc):
                sentence.predicted_ner = predictions

        if self._loss_weights["relation"] > 0:
            for predictions, sentence in zip(output_dict["relation"]["predictions"], doc):
                sentence.predicted_relations = predictions

        return doc

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_ner.keys()) + list(metrics_relation.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_ner.items()) +
                           list(metrics_relation.items()))

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
