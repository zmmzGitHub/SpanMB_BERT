import logging
from typing import Any, Dict, List, Optional, Callable

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util, RegularizerApplicator
from allennlp.modules import TimeDistributed, FeedForward
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor

from spanmb.training.relation_metrics import RelationMetrics
from spanmb.models.entity_beam_pruner import Pruner
from spanmb.data.dataset_readers import document
from spanmb.models.span_extractor import MaxSpanExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO(dwadden) add tensor dimension comments.
# TODO(dwadden) Different sentences should have different number of relation candidates depending on
# length.
class RelationExtractor(Model):
    """
    Relation extraction module of DyGIE model.
    """
    # TODO(dwadden) add option to make `mention_feedforward` be the NER tagger.

    def __init__(self,
                 vocab: Vocabulary,
                 make_feedforward: Callable,
                 token_emb_dim: int,
                 span_emb_dim: int,
                 feature_size: int,
                 spans_per_word: float,
                 rel_prop: int = 0,
                 rel_prop_dropout_A: float = 0.0,
                 rel_prop_dropout_f: float = 0.0,
                 positive_label_weight: float = 1.0,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._endpoint_span_extractor = EndpointSpanExtractor(
            input_dim=token_emb_dim,
            combination="x,y",
            num_width_embeddings=10,
            span_width_embedding_dim=feature_size,
            bucket_widths=True)
        self._max_span_extractor = MaxSpanExtractor(
            input_dim=token_emb_dim
        )
        # max pooling on span and span width feature
        # self._max_span_extractor = MaxSpanExtractor(
        #     input_dim=token_emb_dim,
        #     num_width_embeddings=10,
        #     span_width_embedding_dim=feature_size,
        #     bucket_widths=True
        # )
        context_span_dim = self._endpoint_span_extractor.get_output_dim() +\
                           self._max_span_extractor.get_output_dim()
        self.rel_prop = rel_prop
        self._rel_namespaces = [entry for entry in vocab.get_namespaces() if "relation_labels" in entry]
        self._rel_n_labels = {name: vocab.get_vocab_size(name) for name in self._rel_namespaces}

        self._mention_pruners = torch.nn.ModuleDict()
        self._relation_feedforwards = torch.nn.ModuleDict()
        self._relation_scorers = torch.nn.ModuleDict()
        self._A_networks = torch.nn.ModuleDict()
        self._f_networks = torch.nn.ModuleDict()
        self._relation_metrics = {}

        for namespace in self._rel_namespaces:
            mention_feedforward = make_feedforward(input_dim=span_emb_dim)
            feedforward_scorer = torch.nn.Sequential(
                TimeDistributed(mention_feedforward),
                TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
            self._mention_pruners[namespace] = Pruner(feedforward_scorer)
            relation_scorer_dim = 2 * span_emb_dim + context_span_dim
            relation_feedforward = make_feedforward(input_dim=relation_scorer_dim)
            self._relation_feedforwards[namespace] = relation_feedforward
            relation_scorer = torch.nn.Linear(
                relation_feedforward.get_output_dim(), self._rel_n_labels[namespace])
            self._relation_scorers[namespace] = relation_scorer

            self._relation_metrics[namespace] = RelationMetrics()

            # Relation Propagation
            hidden_dim = span_emb_dim
            self._A_networks[namespace] = torch.nn.Linear(self._rel_n_labels[namespace], hidden_dim)
            self._f_networks[namespace] = FeedForward(input_dim=2 * hidden_dim,
                                                     num_layers=1,
                                                     hidden_dims=hidden_dim,
                                                     activations=torch.nn.Sigmoid(),
                                                     dropout=rel_prop_dropout_f)
        self._spans_per_word = spans_per_word
        self._active_namespace = None

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    def compute_representations(self,  # type: ignore
                                spans: torch.IntTensor,
                                span_mask,
                                span_embeddings,  # TODO(dwadden) add type.
                                text_embeddings,
                                sentence_lengths,
                                metadata) -> Dict[str, torch.Tensor]:
        self._active_namespace = f"{metadata.dataset}__relation_labels"

        (top_span_embeddings, top_span_mention_scores,
         num_spans_to_keep, top_span_mask,
         top_span_indices, top_spans) = self._prune_spans(
            spans, span_mask, span_embeddings, sentence_lengths)

        context_embeddings = self._context_representation_between_entity_2(
            top_spans, top_span_mask, text_embeddings)

        relation_scores = self._compute_relation_scores(
            self._compute_span_pair_embeddings(top_span_embeddings, context_embeddings), top_span_mention_scores)

        output_dict = {"top_spans": top_spans,
                       "top_span_embeddings": top_span_embeddings,
                       # "top_span_embeddings_rel": top_span_embeddings_rel,
                       "top_span_mention_scores": top_span_mention_scores,
                       "relation_scores": relation_scores,
                       "num_spans_to_keep": num_spans_to_keep,
                       "top_span_indices": top_span_indices,
                       "top_span_mask": top_span_mask,
                       # "ner_scorer": self._ner_scorer,
                       "loss": 0}

        return output_dict

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask,
                span_embeddings,  # TODO(dwadden) add type.
                sentence_lengths,
                relation_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        TODO(dwadden) Write documentation.
        """
        self._active_namespace = f"{metadata.dataset}__relation_labels"

        (top_span_embeddings, top_span_mention_scores,
         num_spans_to_keep, top_span_mask,
         top_span_indices, top_spans) = self._prune_spans(
             spans, span_mask, span_embeddings, sentence_lengths)

        relation_scores = self._compute_relation_scores(
            self._compute_span_pair_embeddings(top_span_embeddings), top_span_mention_scores)

        prediction_dict, predictions = self.predict(top_spans.detach().cpu(),
                                                    relation_scores.detach().cpu(),
                                                    num_spans_to_keep.detach().cpu(),
                                                    metadata)

        output_dict = {"predictions": predictions}

        # Evaluate loss and F1 if labels were provided.
        if relation_labels is not None:
            # Compute cross-entropy loss.
            gold_relations = self._get_pruned_gold_relations(
                relation_labels, top_span_indices, top_span_mask)

            cross_entropy = self._get_cross_entropy_loss(relation_scores, gold_relations)

            # Compute F1.
            assert len(prediction_dict) == len(metadata)  # Make sure length of predictions is right.
            relation_metrics = self._relation_metrics[self._active_namespace]
            relation_metrics(prediction_dict, metadata)

            output_dict["loss"] = cross_entropy
        return output_dict

    def _prune_spans(self, spans, span_mask, span_embeddings, sentence_lengths):
        # Prune
        num_spans = spans.size(1)  # Max number of spans for the minibatch.

        # Keep different number of spans for each minibatch entry.
        num_spans_to_keep = torch.ceil(sentence_lengths.float() * self._spans_per_word).long()

        pruner = self._mention_pruners[self._active_namespace]
        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores, num_spans_kept) = pruner(
             span_embeddings, span_mask, num_spans_to_keep)

        top_span_mask = top_span_mask.unsqueeze(-1)

        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        return top_span_embeddings, top_span_mention_scores, num_spans_to_keep, top_span_mask, top_span_indices, top_spans

    def relation_propagation(self, output_dict):
        relation_scores = output_dict["relation_scores"]
        top_span_embeddings = output_dict["top_span_embeddings"]
        var = output_dict["top_span_mask"]
        # [batch_size, span_num, span_num]*[batch_size, span_num, span_num]: mask for relation
        top_span_mask_tensor = (var.repeat(1, 1, var.shape[1]) * var.view(var.shape[0], 1, var.shape[1]).repeat(1, var.shape[1], 1)).float()
        span_num = relation_scores.shape[1]
        normalization_factor = var.view(var.shape[0], span_num).sum(dim=1).float()
        pruner = self._mention_pruners[self._active_namespace]
        for t in range(self.rel_prop):
            # TODO(Ulme) There is currently an implicit assumption that the null label is in the 0-th index.
            # Come up with how to deal with this
            relation_scores = F.relu(relation_scores[:, :, :, 1:], inplace=False)
            relation_embeddings = self._A_networks[self._active_namespace](relation_scores)
            relation_embeddings = (relation_embeddings.transpose(3, 2).transpose(2, 1).transpose(1, 0) * top_span_mask_tensor).transpose(0, 1).transpose(1, 2).transpose(2, 3)
            entity_embs = torch.sum(relation_embeddings.transpose(2, 1).transpose(1, 0) * top_span_embeddings, dim=0)
            entity_embs = (entity_embs.transpose(0, 2) / normalization_factor).transpose(0, 2)
            f_network_input = torch.cat([top_span_embeddings, entity_embs], dim=-1)
            f_weights = self._f_networks[self._active_namespace](f_network_input)
            top_span_embeddings = f_weights * top_span_embeddings + (1.0 - f_weights) * entity_embs
            relation_scores = self._compute_relation_scores(
                self._compute_span_pair_embeddings(top_span_embeddings), pruner._scorer(top_span_embeddings))

        output_dict["relation_scores"] = relation_scores
        output_dict["top_span_embeddings"] = top_span_embeddings
        # output_dict["top_span_embeddings_rel"] = top_span_embeddings_rel
        return output_dict

    def predict_labels(self, relation_labels, output_dict, metadata):
        relation_scores = output_dict["relation_scores"]
        top_spans = output_dict["top_spans"]
        num_spans_to_keep = output_dict["num_spans_to_keep"]
        top_predicted_ner = output_dict["top_ner_predicted_labels"]
        # [batch_size,span_num, span_num]
        # ner_mask = (top_predicted_ner > 0).float()
        # ner_mask_tiled_x = ner_mask.unsqueeze(1).repeat(1, top_spans.size(1), 1)
        # ner_mask_tiled_y = ner_mask.unsqueeze(2).repeat(1, 1, top_spans.size(1))
        # ner_mask_tiled = ner_mask_tiled_x * ner_mask_tiled_y
        # # relations on spans which is predicted as entities
        # relation_scores = relation_scores * ner_mask_tiled.unsqueeze(-1)
        top_gold_ner = output_dict["top_ner_gold_labels"]
        prediction_dict, predictions = self.predict(top_spans.detach().cpu(),
                                                    relation_scores.detach().cpu(),
                                                    num_spans_to_keep.detach().cpu(),
                                                    top_predicted_ner.detach().cpu(),
                                                    top_gold_ner.detach().cpu(),
                                                    metadata)
        # output_dict includes relation predictions and predicted ner labels of span pair
        # output_dict["predicted_ners"] is for span pair ner labels
        output_dict["predictions"] = predictions

        # Evaluate loss and F1 if labels were provided.
        if relation_labels is not None:
            # Compute cross-entropy loss.
            gold_relations = self._get_pruned_gold_relations(
                relation_labels, output_dict["top_span_indices"], output_dict["top_span_mask"])

            cross_entropy = self._get_cross_entropy_loss(relation_scores, gold_relations)

            # Compute F1.
            assert len(prediction_dict["preds_dict"]) == len(metadata)  # Make sure length of predictions is right.
            # separate metric with each label
            relation_metrics = self._relation_metrics[self._active_namespace]
            relation_metrics(prediction_dict, metadata)

            output_dict["loss"] = cross_entropy
        return output_dict


    def predict(self, top_spans, relation_scores, num_spans_to_keep, top_predicted_ner, top_gold_ner, metadata):
        preds_dict = []
        predictions = []
        predicted_ners = []
        gold_ners = []
        zipped = zip(top_spans, relation_scores, num_spans_to_keep, top_predicted_ner, top_gold_ner, metadata)

        for top_spans_sent, relation_scores_sent, num_spans_sent, top_predicted_ner_sent, top_gold_ner_sent, sentence in zipped:
            pred_dict_sent, predictions_sent, predicted_ner_sent, gold_ner_sent = self._predict_sentence(
                top_spans_sent, relation_scores_sent, num_spans_sent, top_predicted_ner_sent, top_gold_ner_sent, sentence)
            preds_dict.append(pred_dict_sent)
            predictions.append(predictions_sent)
            predicted_ners.append(predicted_ner_sent)
            gold_ners.append(gold_ner_sent)

        predicted_dict = {"preds_dict": preds_dict,
                          "predicted_ners": predicted_ners,
                          "gold_ners": gold_ners}

        return predicted_dict, predictions

    def _predict_sentence(self, top_spans, relation_scores, num_spans_to_keep, top_predicted_ner, top_gold_ner, sentence):
        keep = num_spans_to_keep.item()
        top_spans = [tuple(x) for x in top_spans.tolist()]

        # Iterate over all span pairs and labels. Record the span if the label isn't null.
        predicted_scores_raw, predicted_labels = relation_scores.max(dim=-1)
        softmax_scores = F.softmax(relation_scores, dim=-1)
        predicted_scores_softmax, _ = softmax_scores.max(dim=-1)
        predicted_labels -= 1  # Subtract 1 so that null labels get -1.

        keep_mask = torch.zeros([len(top_spans), len(top_spans)])
        keep_mask[:keep, :keep] = 1
        # entity has no relation with its self.
        for i in range(keep):
            keep_mask[i, i] = 0
        keep_mask = keep_mask.bool()

        ix = (predicted_labels >= 0) & keep_mask

        res_dict = {}
        predictions = []
        predicted_ner_list = []
        gold_ner_list = []

        for i, j in ix.nonzero(as_tuple=False):
            span_1 = top_spans[i]
            span_2 = top_spans[j]
            label = predicted_labels[i, j].item()
            raw_score = predicted_scores_raw[i, j].item()
            softmax_score = predicted_scores_softmax[i, j].item()

            label_name = self.vocab.get_token_from_index(label, namespace=self._active_namespace)
            res_dict[(span_1, span_2)] = label_name
            list_entry = (span_1[0], span_1[1], span_2[0], span_2[1], label_name, raw_score, softmax_score)
            predictions.append(document.PredictedRelation(list_entry, sentence, sentence_offsets=True))
            ner_namespace = self._active_namespace.split('_')[0] + '__ner_labels'
            span_pair_predicted_ner_type = (
                self.vocab.get_token_from_index(top_predicted_ner[i].item(), namespace=ner_namespace),
                self.vocab.get_token_from_index(top_predicted_ner[j].item(), namespace=ner_namespace))
            span_pair_gold_ner_type = (
                self.vocab.get_token_from_index(top_gold_ner[i].item(), namespace=ner_namespace),
                self.vocab.get_token_from_index(top_gold_ner[j].item(), namespace=ner_namespace))
            predicted_ner_list.append(span_pair_predicted_ner_type)
            gold_ner_list.append(span_pair_gold_ner_type)
        assert len(res_dict) == len(predictions)

        return res_dict, predictions, predicted_ner_list, gold_ner_list

    # TODO(dwadden) This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._relation_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            to_update = {f"{prefix}_precision": precision,
                         f"{prefix}_recall": recall,
                         f"{prefix}_f1": f1}
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__relation_{name}"] = sum(values) / len(values) if values else 0
            res.update(res_avg)

        return res

    def _context_representation_between_entity(self, top_spans, top_span_mask, text_embeddings):
        b_size = top_spans.size(0)
        span_num = top_spans.size(1)
        ctx_span = []
        ctx_span_loc = []
        for k in range(b_size):
            one_seq_ctx_span = []
            one_seq_ctx_loc = []
            for i in range(span_num):
                for j in range(span_num):
                    if top_span_mask[k, i] == 0 or top_span_mask[k, j] == 0:
                        break
                    # up
                    if i < j and top_spans[k, i, 1] + 1 < top_spans[k, j, 0]:
                        one_seq_ctx_span.append(torch.tensor([top_spans[k, i, 1] + 1, top_spans[k, j, 0] - 1],
                                                             device=top_spans.device, dtype=top_spans.dtype))
                        one_seq_ctx_loc.append([i, j])
            ctx_span.append(one_seq_ctx_span)
            ctx_span_loc.append(one_seq_ctx_loc)
        seq_span_num = [len(seq) for seq in ctx_span]
        max_span_num = max(seq_span_num)
        # ctx_span_mask = util.get_mask_from_sequence_lengths(torch.Tensor(seq_span_num, device=top_spans.device), max_span_num).long()
        ctx_span_tensor = top_spans.new_zeros([b_size, max_span_num, 2])
        for k in range(b_size):
            for i in range(seq_span_num[k]):
                ctx_span_tensor[k, i, :] = ctx_span[k][i]
        # After getting all span embeddings, construct context representation between entities
        rel_ctx_emb = text_embeddings.new_zeros([b_size, span_num, span_num, text_embeddings.size(-1)])
        ctx_emb = self._context_mean_embeddings(text_embeddings, ctx_span_tensor)
        for k in range(b_size):
            for i in range(seq_span_num[k]):
                x = ctx_span_loc[k][i][0]
                y = ctx_span_loc[k][i][1]
                rel_ctx_emb[k, x, y, :] = ctx_emb[k, i, :]
                rel_ctx_emb[k, y, x, :] = ctx_emb[k, i, :]

        return rel_ctx_emb

    def _context_mean_embeddings(self, text_embeddings, ctx_span):
        # [batch_size, span_num, max_span_width, emb_size]
        span_embeddings, span_mask = util.batched_span_select(text_embeddings, ctx_span)
        # set unmask embeddings to- 0
        span_embeddings = span_embeddings * span_mask.unsqueeze(-1).float()
        # [batch_size, span_num, emb_size]
        span_avg_embeddings = torch.mean(span_embeddings, 2)  # span_mask
        return span_avg_embeddings

    def _context_max_embeddings(self, text_embeddings, ctx_span):
        # [batch_size, span_num, max_span_width, emb_size]
        span_embeddings, span_mask = util.batched_span_select(text_embeddings, ctx_span)
        # set unmask embeddings to- 0
        span_embeddings = span_embeddings * span_mask.unsqueeze(-1).float()
        # [batch_size, span_num, emb_size]
        span_max_embeddings = torch.max(span_embeddings, 2)[0]  # span_mask
        return span_max_embeddings

    def _context_representation_between_entity_2(self, top_spans, top_span_mask, text_embeddings):
        b_size = top_spans.size(0)
        num_span = top_spans.size(1)
        top_span_x = top_spans.unsqueeze(1)
        top_span_y = top_spans.unsqueeze(2)
        top_span_x_tiled = top_span_x.repeat(1, num_span, 1, 1)
        top_span_y_tiled = top_span_y.repeat(1, 1, num_span, 1)

        top_context_len = top_span_x_tiled[:, :, :, 0] - top_span_y_tiled[:, :, :, 1]
        top_span_mask = top_span_mask.squeeze(-1)
        span_mask_x = top_span_mask.unsqueeze(1)
        span_mask_y = top_span_mask.unsqueeze(2)
        span_mask_x_tiled = span_mask_x.repeat(1, num_span, 1)
        span_mask_y_tiled = span_mask_y.repeat(1, 1, num_span)
        span_mask_tiled = span_mask_x_tiled * span_mask_y_tiled

        top_context_masku = (top_context_len > 1).float() * span_mask_tiled  # expand span_mask as span tiled
        top_context_maskl = top_context_masku.transpose(1, 2)
        top_context_span = torch.cat([top_span_y_tiled[:, :, :, 1:] + 1,
                                      top_span_x_tiled[:, :, :, 0:1] - 1], -1)
        top_context_spanu = top_context_span * top_context_masku.unsqueeze(-1)
        top_context_spanl = top_context_span.transpose(1, 2) * top_context_maskl.unsqueeze(-1)
        # Exchange span_start and span_end
        top_context_spanl = torch.cat([top_context_spanl[:, :, :, 1:],
                                       top_context_spanl[:, :, :, 0:1]], -1)
        top_context_span_f = top_context_spanu + top_context_spanl
        top_context_span_f = top_context_span_f.long()
        # dis_ctx_span = top_context_span_f[:, :, :, 0] - top_context_span_f[:, :, :, 1]
        # max_dis = dis_ctx_span.max()
        # print("max_dis: ", max_dis)

        rel_ctx_emb = text_embeddings.new_zeros([b_size, num_span, num_span, self._endpoint_span_extractor.get_output_dim()])
        rel_ctx_emb_max = text_embeddings.new_zeros([b_size, num_span, num_span, self._max_span_extractor.get_output_dim()])
        for i in range(num_span):
            rel_ctx_emb_max[:, i, :, :] = self._max_span_extractor(text_embeddings, top_context_span_f[:, i])
            rel_ctx_emb[:, i, :, :] = self._endpoint_span_extractor(text_embeddings, top_context_span_f[:, i])

        return torch.cat([rel_ctx_emb_max, rel_ctx_emb], -1)
        # return rel_ctx_emb

    @staticmethod
    def _compute_span_pair_embeddings(top_span_embeddings: torch.FloatTensor, context_embeddings):
        """
        TODO(dwadden) document me and add comments.
        """
        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep, embedding_size)
        num_candidates = top_span_embeddings.size(1)

        embeddings_1_expanded = top_span_embeddings.unsqueeze(2)
        embeddings_1_tiled = embeddings_1_expanded.repeat(1, 1, num_candidates, 1)

        embeddings_2_expanded = top_span_embeddings.unsqueeze(1)
        embeddings_2_tiled = embeddings_2_expanded.repeat(1, num_candidates, 1, 1)

        # similarity_embeddings = embeddings_1_expanded * embeddings_2_expanded
        # relation context (context between candidate entity pairs)

        pair_embeddings_list = [embeddings_1_tiled, context_embeddings, embeddings_2_tiled]      #, similarity_embeddings
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

        return pair_embeddings

    def _compute_relation_scores(self, pairwise_embeddings, top_span_mention_scores):
        relation_feedforward = self._relation_feedforwards[self._active_namespace]
        relation_scorer = self._relation_scorers[self._active_namespace]

        batch_size = pairwise_embeddings.size(0)
        max_num_spans = pairwise_embeddings.size(1)
        feature_dim = relation_feedforward.input_dim

        embeddings_flat = pairwise_embeddings.view(-1, feature_dim)

        relation_projected_flat = relation_feedforward(embeddings_flat)
        relation_scores_flat = relation_scorer(relation_projected_flat)

        relation_scores = relation_scores_flat.view(batch_size, max_num_spans, max_num_spans, -1)

        # Add the mention scores for each of the candidates.

        relation_scores += (top_span_mention_scores.unsqueeze(-1) +
                            top_span_mention_scores.transpose(1, 2).unsqueeze(-1))

        shape = [relation_scores.size(0), relation_scores.size(1), relation_scores.size(2), 1]
        dummy_scores = relation_scores.new_zeros(*shape)

        relation_scores = torch.cat([dummy_scores, relation_scores], -1)
        return relation_scores

    @staticmethod
    def _get_pruned_gold_relations(relation_labels, top_span_indices, top_span_masks):
        """
        Loop over each slice and get the labels for the spans from that slice.
        All labels are offset by 1 so that the "null" label gets class zero. This is the desired
        behavior for the softmax. Labels corresponding to masked relations keep the label -1, which
        the softmax loss ignores.
        """
        # TODO(dwadden) Test and possibly optimize.
        relations = []

        zipped = zip(relation_labels, top_span_indices, top_span_masks.bool())
        for sliced, ixs, top_span_mask in zipped:
            entry = sliced[ixs][:, ixs].unsqueeze(0)
            mask_entry = top_span_mask & top_span_mask.transpose(0, 1).unsqueeze(0)
            entry[mask_entry] += 1
            entry[~mask_entry] = -1
            relations.append(entry)

        return torch.cat(relations, dim=0)

    def _get_cross_entropy_loss(self, relation_scores, relation_labels):
        """
        Compute cross-entropy loss on relation labels. Ignore diagonal entries and entries giving
        relations between masked out spans.
        """
        # Need to add one for the null class.
        n_labels = self._rel_n_labels[self._active_namespace] + 1
        scores_flat = relation_scores.view(-1, n_labels)
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = relation_labels.view(-1)
        # Compute cross-entropy loss.
        loss = self._loss(scores_flat, labels_flat)
        return loss
