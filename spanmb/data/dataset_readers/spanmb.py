import logging
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import json
import pickle as pkl
import warnings
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (ListField, TextField, SpanField, MetadataField,
                                  SequenceLabelField, AdjacencyField, LabelField, ArrayField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from spanmb.data.fields.adjacency_field_assym import AdjacencyFieldAssym
from spanmb.data.dataset_readers.document import Document, Sentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SpanMBDataException(Exception):
    pass


@DatasetReader.register("spanmb")
class SpanMBReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Loop over the documents.
            doc_text = json.loads(line)     # str to dict
            instance = self.text_to_instance(doc_text)
            yield instance

    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width

    def _process_ner(self, span_tuples, sent):
        ner_labels = [""] * len(span_tuples)
        if sent.nested_entity is not None:
            ner_nested = [0] * len(span_tuples)
            for span, is_nested in zip(sent.ner_dict.keys(), sent.nested_entity):
                if self._too_long(span):
                    continue
                ix = span_tuples.index(span)
                ner_labels[ix] = sent.ner_dict[span]
                ner_nested[ix] = is_nested
            return ner_labels, ner_nested
        else:
            for span, label in sent.ner_dict.items():
                if self._too_long(span):
                    continue
                ix = span_tuples.index(span)
                ner_labels[ix] = label
            return ner_labels

    def _process_relations(self, span_tuples, sent):
        relations = []
        relation_indices = []

        # Loop over the gold spans. Look up their indices in the list of span tuples and store
        # values.
        for (span1, span2), label in sent.relation_dict.items():
            # If either span is beyond the max span width, skip it.
            if self._too_long(span1) or self._too_long(span2):
                continue
            ix1 = span_tuples.index(span1)
            ix2 = span_tuples.index(span2)
            relation_indices.append((ix1, ix2))
            relations.append(label)

        return relations, relation_indices

    def _process_sentence(self, sent: Sentence, dataset: str):
        # Get the sentence text and define the `text_field`.
        sentence_text = [self._normalize_word(word) for word in sent.text]
        text_field = TextField([Token(word) for word in sentence_text], self._token_indexers)

        # Enumerate spans.
        spans = []
        for start, end in enumerate_spans(sentence_text, max_span_width=self._max_span_width):
            # span_text = "".join(sentence_text[i] for i in range(start, end + 1))
            # # bacteria biotope text contains space token. Remove span beginning or ending with space token
            # if not (span_text.startswith(" ") or span_text.endswith(" ")):
            spans.append(SpanField(start, end, text_field))
        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]

        # Convert data to fields.
        # NOTE: The `ner_labels` and `coref_labels` would ideally have type
        # `ListField[SequenceLabelField]`, where the sequence labels are over the `SpanField` of
        # `spans`. But calling `as_tensor_dict()` fails on this specific data type. Matt G
        # recognized that this is an AllenNLP API issue and suggested that represent these as
        # `ListField[ListField[LabelField]]` instead.
        fields = {}
        fields["text"] = text_field
        fields["spans"] = span_field
        if sent.ner is not None:
            if sent.nested_entity is None:
                ner_labels = self._process_ner(span_tuples, sent)
                fields["ner_labels"] = ListField(
                    [LabelField(entry, label_namespace=f"{dataset}__ner_labels")
                     for entry in ner_labels])
            else:
                ner_labels, ner_nested = self._process_ner(span_tuples, sent)
                fields["ner_labels"] = ListField(
                    [LabelField(entry, label_namespace=f"{dataset}__ner_labels")
                     for entry in ner_labels])
                fields["ner_nested"] = ArrayField(np.array(ner_nested))

        if sent.relations is not None:
            relation_labels, relation_indices = self._process_relations(span_tuples, sent)
            fields["relation_labels"] = AdjacencyField(
                indices=relation_indices, sequence_field=span_field, labels=relation_labels,
                label_namespace=f"{dataset}__relation_labels")

        return fields

    def _process_sentence_fields(self, doc: Document):
        # Process each sentence.
        sentence_fields = [self._process_sentence(sent, doc.dataset) for sent in doc.sentences]

        # Make sure that all sentences have the same set of keys.
        first_keys = set(sentence_fields[0].keys())
        for entry in sentence_fields:
            if set(entry.keys()) != first_keys:
                raise SpanMBDataException(
                    f"Keys do not match across sentences for document {doc.doc_key}.")

        # For each field, store the data from all sentences together in a ListField.
        fields = {}
        keys = sentence_fields[0].keys()
        for key in keys:
            this_field = ListField([sent[key] for sent in sentence_fields])
            fields[key] = this_field

        return fields

    @overrides
    def text_to_instance(self, doc_text: Dict[str, Any]):
        """
        Convert a Document object into an instance.
        """
        doc = Document.from_json(doc_text)

        # Make sure there are no single-token sentences; these break things.
        sent_lengths = [len(x) for x in doc.sentences]
        # ner_num = sum([len(x.ner) for x in doc.sentences])
        if min(sent_lengths) < 2:
            msg = (f"Document {doc.doc_key} has a sentence with a single token or no tokens. "
                   "This may break the modeling code.")
            warnings.warn(msg)

        fields = self._process_sentence_fields(doc)
        fields["metadata"] = MetadataField(doc)

        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f, protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
