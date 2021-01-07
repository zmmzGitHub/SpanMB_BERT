from overrides import overrides

from allennlp.training.metrics.metric import Metric

from spanmb.training.f1 import compute_f1


class RelationMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    """
    def __init__(self):
        self.reset()

    # TODO(dwadden) This requires decoding because the dataset reader gets rid of gold spans wider
    # than the span width. So, I can't just compare the tensor of gold labels to the tensor of
    # predicted labels.
    @overrides
    def __call__(self, predicted_dict, metadata_list):
        predicted_relation_list = predicted_dict["preds_dict"]
        predicted_ner_list = predicted_dict["predicted_ners"]
        gold_ner_list = predicted_dict["gold_ners"]

        for predicted_relations, predicted_ner_pairs, gold_ner_pairs, metadata in zip(predicted_relation_list, predicted_ner_list, gold_ner_list, metadata_list):
            gold_relations = metadata.relation_dict
            if metadata.rel_num is not None:
                self._total_gold += metadata.rel_num   # eval
            else:
                self._total_gold += len(gold_relations)   # train

            self._total_predicted += len(predicted_relations)
            for span_pair, predicted_ner, gold_ner in zip(predicted_relations.keys(), predicted_ner_pairs, gold_ner_pairs):
                label = predicted_relations[span_pair]
                if span_pair in gold_relations and predicted_ner == gold_ner and gold_relations[span_pair] == label:
                    self._total_matched += 1


    @overrides
    def get_metric(self, reset=False):
        precision, recall, f1 = compute_f1(self._total_predicted, self._total_gold, self._total_matched)
        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1

    @overrides
    def reset(self):
        self._total_gold = 0
        self._total_predicted = 0
        self._total_matched = 0
