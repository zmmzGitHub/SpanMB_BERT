"""
Spot-check results to make sure we kept most entities and relationships.

Some will be dropped due to tokenization / sentence splitting.
"""

import pandas as pd
from collections import Counter

from spanmb.data.dataset_readers.document import Dataset


def spot_check_fold(fold):
    print(f"Checking {fold}.")
    fname = f"data/bb/processed_data/json/{fold}.jsonl"
    data = Dataset.from_jsonl(fname)

    f_entity = f"data/bb/processed_data/merge_data/BB_{fold}/bb_{fold}_entities.tsv"
    entities = pd.read_table(f_entity, header=None)
    entities.columns = ["doc_key", "entity_id", "label", "char_start_end_ids", "text"]

    f_relation = f"data/bb/processed_data/merge_data/BB_{fold}/bb_{fold}_relations.tsv"
    relations = pd.read_table(f_relation, header=None)
    relations.columns = ["doc_key", "rel_id", "label", "e_t1", "e_t2"]

    res = []

    for entry in data:
        counts = Counter()
        expected_entities = entities.query(f"doc_key == '{entry.doc_key}'")
        expected_relations = relations.query(f"doc_key == '{entry.doc_key}'")
        for sent in entry:
            counts["found_entities"] += len(sent.ner)
            counts["found_relations"] += len(sent.relations)

        counts["expected_entities"] = len(expected_entities)
        counts["expected_relations"] = len(expected_relations)

        counts["doc_key"] = entry.doc_key
        res.append(counts)

    res = pd.DataFrame(res).set_index("doc_key")

    frac_entities = res["found_entities"].sum() / res["expected_entities"].sum()
    frac_relations = res["found_relations"].sum() / res["expected_relations"].sum()

    print(f"Fraction of entities preserved from original file: {frac_entities:0.2f}")
    print(f"Fraction of relations preserved from original file: {frac_relations:0.2f}")


def main():
    for fold in ["train", "dev"]:
        spot_check_fold(fold)


if __name__ == "__main__":
    main()
