from overrides import overrides
from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric

from spanmb.training.f1 import compute_f1

# TODO(dwadden) Need to use the decoded predictions so that we catch the gold examples longer than
# the span boundary.

class NERMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """
    def __init__(self, number_of_classes: int, none_label: int=0):
        self.number_of_classes = number_of_classes
        self.none_label = none_label
        self.reset()

    @overrides
    def __call__(self, predictions, gold_ner, mask):
        predictions = predictions.cpu()
        gold_labels = gold_ner["ner_labels"].cpu()
        metadata_list = gold_ner["metadata_list"]
        for metadata in metadata_list:
            if metadata.ner_num is not None:
                self._ner_gold_num += metadata.ner_num

        mask = mask.cpu()
        for i in range(self.number_of_classes):
            if i == self.none_label:
                continue
            self._true_positives += ((predictions==i)*(gold_labels==i)*mask.bool()).sum().item()
            self._false_positives += ((predictions==i)*(gold_labels!=i)*mask.bool()).sum().item()
            self._true_negatives += ((predictions!=i)*(gold_labels!=i)*mask.bool()).sum().item()
            self._false_negatives += ((predictions!=i)*(gold_labels==i)*mask.bool()).sum().item()

    @overrides
    def get_metric(self, reset=False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        predicted = self._true_positives + self._false_positives
        if self._ner_gold_num != 0:
            gold = self._ner_gold_num
        else:
            gold = self._true_positives + self._false_negatives
        matched = self._true_positives
        precision, recall, f1_measure = compute_f1(predicted, gold, matched)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1_measure

    @overrides
    def reset(self):
        self._true_positives = 0
        self._false_positives = 0
        self._true_negatives = 0
        self._false_negatives = 0
        self._ner_gold_num = 0
