import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm
import json
import os
import numpy as np
import re
from itertools import accumulate

class NToken:
    def __init__(self, idx, i, text):
        self.idx = idx
        self.i = i
        self.text = text

    def __len__(self):
        return len(self.text)


class NSpan:
    def __init__(self, start_char_id, end_char_id, toks, text):
        self.start_char = start_char_id
        self.end_char = end_char_id
        self.toks = toks
        self.text = text

    def __iter__(self):
        return iter(self.toks)


# different models split sentences differently, so found different number of relations within sentence.
nlp = spacy.load("en_core_sci_md")     # tokenizer, "en_core_sci_md" is bigger
add_id = [0]

####################

# Process entities and relations for a given abstract.


def get_entities_in_sent(sent, entities):
    start, end = sent.start_char, sent.end_char
    span_ids = entities["char_start_end_ids"].values    # return ndarray of id string
    # print("span_ids: ", type(span_ids), span_ids)
    start_ids = []
    end_ids = []
    for span_id in span_ids:
        # may be more than one span
        start_ids.append(int(span_id[:span_id.index(' ')]))
        end_ids.append(int(span_id[span_id.rindex(' ')+1:]))
        # print('span: ', span_id, ' start: ', start_ids[-1], ' end: ', end_ids[-1])

    start_ids = np.array(start_ids)
    end_ids = np.array(end_ids)
    start_ok = start_ids >= start
    end_ok = end_ids <= end
    keep = start_ok & end_ok
    res = entities[keep]
    # print("sent start: ", start, " sent end: ", end, " keep len: ", len(res))
    return res


def align_one(sent, row):
    ids = re.split("\s|;", row['char_start_end_ids'])
    start_ids = []
    end_ids = []
    for i in range(len(ids)):
        if i % 2 == 0:
            start_ids.append(int(ids[i]))
        else:
            end_ids.append(int(ids[i]))

    start_tok = []
    end_tok = []

    for tok in sent:
        if tok.idx in start_ids:
            start_tok.append(tok)
        if tok.idx + len(tok) in end_ids:
            end_tok.append(tok)

    if len(start_tok) < len(start_ids) or len(end_tok) < len(end_ids):
        print("sent: ", sent.text, " start_ids: ", start_ids, " end_ids: ", end_ids, " start_tok: ", [t.text for t in start_tok],
              "end_tok: ", [t.text for t in end_tok])
        print("sent toks: ", [t.text for t in sent])
        return None
    else:
        expected = []
        start_tok_id = []
        end_tok_id = []
        for j in range(len(start_tok)):   #
            expected.append(sent.text[start_tok[j].idx - sent.start_char:end_tok[j].idx - sent.start_char+len(end_tok[j])])
            start_tok_id.append(start_tok[j].i)
            end_tok_id.append(end_tok[j].i)

        if '\xa0' in row.text:   # latin 1 to utf-8
            row.text = row.text.replace('\xa0', ' ')
        expected_str = ' '.join(expected)
        if '   ' in expected_str:    # ' ' in center of expected, three space join
            expected_str = expected_str.replace('   ', '  ')
        if expected_str != row.text:
            print("entity index wrong:\n", "row text : ", row.text, ' expected: ', expected)
            print("sent: ", sent.text, " start_ids: ", start_ids, " end_ids: ", end_ids, " start_tok: ", start_tok,
                  "end_tok: ", end_tok)
            print("sent toks: ", [t.text for t in sent])
            # return row["doc_key"]
            raise Exception("Entity mismatch")
        return (start_tok_id, end_tok_id, row["label"])    # replace start_ids with start_tok_id

def align_entities(sent, entities_sent):
    aligned_entities = {}
    continuous_entities = {}
    missed_entities = {}
    for i, row in entities_sent.iterrows():   # entity in sent
        aligned = align_one(sent, row)
        if aligned is not None:
            aligned_entities[row["entity_id"]] = aligned
            if len(aligned[0]) == 1:
                continuous_entities[row["entity_id"]] = (aligned[0][0], aligned[1][0], aligned[2])
        else:
            missed_entities[row["entity_id"]] = aligned

    # sentence re-tokenization

    return aligned_entities, continuous_entities, missed_entities

def is_nested_entities(entities_list):
    is_nested = [0] * len(entities_list)
    # judge if nested entities
    for i in range(len(entities_list)):
        for j in range(len(entities_list)):
            if i != j:
                ent_i = entities_list[i]
                ent_j = entities_list[j]
                if ent_j[0] <= ent_i[0] and ent_i[1] <= ent_j[1] or ent_i[0] <= ent_j[0] and ent_j[1] <= ent_i[1]:
                    is_nested[i] = 1
                    break
    return is_nested

def format_relations(relations):
    # Convert to dict.
    res = {}
    for _, row in relations.iterrows():
        ent1 = row["e_t1"].split(':')[-1]
        ent2 = row["e_t2"].split(':')[-1]    # divide label and entity,
        key = (ent1, ent2)
        res[key] = row["label"]

    return res


def get_relations_in_sent(aligned, continuous, relations):
    res = []
    keys = set()
    rel_missed = {}
    rel_discon_ent = {}
    # Loop over the relations, and keep the ones relating entities in this sentences.
    for ents, label in relations.items():
        # Only store relations between continuous entities.
        if ents[0] in aligned and ents[1] in aligned:
            if ents[0] in continuous and ents[1] in continuous:
                keys.add(ents)
                ent1 = continuous[ents[0]]  # id to (entity_start_end_ids, label)
                ent2 = continuous[ents[1]]
                # add entity label in relation
                to_append = ent1[:2] + ent2[:2] + (label,)  # entity_start_end_ids + rel_label
                res.append(to_append)
            else:
                rel_missed[ents] = label    # relations based on discontinuous entities
                rel_discon_ent[ents] = label
        # Relations not in one sentence
        if ents[0] in aligned and ents[1] not in aligned:
            rel_missed[ents] = label

    return res, keys, rel_missed, rel_discon_ent

def re_sent_token(doc, entities):
    all_entity_ids = set()
    for _, entity in entities.iterrows():
        ids = re.split('[\s;]', entity["char_start_end_ids"])
        all_entity_ids |= set(ids)

    # re-sentence: merge and sep
    new_sents_list = []
    sent_number = len(list(doc.sents))
    # print("Before sents number: ", sent_number)
    dump_sent = None
    for sent in doc.sents:
        # print("add_id: ", add_id)
        if not sent.text[0].islower():
            if dump_sent is not None:
                find_end = re.search('[.!?\"\'\s]', dump_sent.text[-1])
                # print("last char", dump_sent.text[-1])
                if find_end:
                    # make sure dump_sent is a sentence
                    new_sents_list.append(dump_sent)
                    dump_sent = re_sent(sent, all_entity_ids)
                else:
                    dump_sent = merge_sent(dump_sent, sent, all_entity_ids)
            else:
                dump_sent = re_sent(sent, all_entity_ids)
        else:
            if sent.text.startswith('hpaJ'):
                new_sents_list.append(dump_sent)
                dump_sent = re_sent(sent, all_entity_ids)
            elif dump_sent is None:
                dump_sent = re_sent(sent, all_entity_ids)
            else:
                dump_sent = merge_sent(dump_sent, sent, all_entity_ids)
        # print("dump_sent: ", dump_sent.text)
    # add last sent
    new_sents_list.append(dump_sent)
    # for sent in new_sents_list:
    #     print(sent.text)
    #     print([tok.text for tok in sent])

    merge_sent_number = len(new_sents_list)
    # print("After sents number: ", merge_sent_number)
    add_id[0] = 0
    return new_sents_list

def merge_sent(sent1, sent2, entity_ids):
    # print("sent1: ", sent1.text, "\nsent2: ", sent2.text)
    # print("sent1 id: ", sent1.end_char, " sent2 id:", sent2.start_char)
    diff_id = sent2.start_char-sent1.end_char
    new_toks = []
    for tok in sent1:     # sent1 is dump_sent
        new_toks.append(tok)
    for tok in sent2:
        new_toks.extend(re_token(tok, entity_ids))
    new_sent = NSpan(sent1.start_char, sent2.end_char, new_toks, sent1.text + ' ' * diff_id + sent2.text)
    return new_sent

def re_sent(sent, entity_ids):
    new_toks = []
    for tok in sent:
        new_toks.extend(re_token(tok, entity_ids))

    new_sent = NSpan(sent.start_char, sent.end_char, new_toks, sent.text)
    return new_sent

def re_token(tok, entity_ids):
    new_toks = []
    # for one_id in entity_ids:
    #     # After splitting, the indices of all tok behind this tok will change
    #     if tok.idx < int(one_id) < tok.idx + len(tok):
    toks = None
    if re.search('([-/_.\[)])', tok.text):
        toks = re.split('([-/_.\[)])', tok.text)
    elif re.search('(\d+)', tok.text):
        toks = re.split('(\d+)', tok.text)  # one special tok
    if toks is not None:
        toks = [tok for tok in toks if tok is not '']
        lens = [0] + list(accumulate([len(t) for t in toks[:-1]]))
        # print('tok: ', tok.text, ' idx: ', tok.idx, ' tok.i: ', tok.i, ' toks: ', toks)
        for i in range(len(toks)):
            t_idx = tok.idx + lens[i]
            ntok = NToken(t_idx, tok.i + i + add_id[0], toks[i])
            # print('tok: ', ntok.text, ' idx: ', ntok.idx, ' tok.i: ', ntok.i)
            new_toks.append(ntok)

        # print("tok: ", tok.text, ' tok split: ', toks)
        add_id[0] += len(toks) - 1

            # break  # only once
    if len(new_toks) == 0:
        new_toks.append(NToken(tok.idx, tok.i + add_id[0], tok.text))

    return new_toks

####################

# Manage a single document and a single fold.

def one_abstract(row, df_entities, df_relations, df_entity_start_ids):
    doc_key = row["doc_key"]
    print(type(row), type(row['doc_key']))
    if 'F' in doc_key:
        # print(doc_key + ' is a para')
        doc = row["abstract"]
    else:
        # print(doc_key + ' is a abstract')
        doc = row["title"] + " " + row["abstract"]
    entities = df_entities.query(f"doc_key == '{doc_key}'")
    relations = format_relations(df_relations.query(f"doc_key == '{doc_key}'"))
    entity_start_id = df_entity_start_ids.query(f"doc_key == '{doc_key}'").iat[0, 1]
    # entity_start_id_pd = df_entity_start_ids.query(f"doc_key == '{doc_key}'")
    # entity_start_ids = [row for _, row in entity_start_id_pd.iterrows()]
    # entity_start_id = entity_start_ids[0]["entity_start_id"]
    # entity_start_id_2 = entity_start_id_pd.at[entity_start_id_pd.index[0], "entity_start_id"]
    # entity_start_id_3 = entity_start_id_pd.iat[0, 1]

    processed = nlp(doc)

    processed_doc = re_sent_token(processed, entities)

    entities_seen = set()
    entities_alignment = set()
    entities_continuous = set()
    entities_no_alignment = set()
    relations_found = set()
    count_continuous = 0
    entity_len_2 = 0
    entity_len_more = 0
    rel_missed_sum = 0
    rel_discon_ent_sum = 0
    nested_entity_sum = 0

    scierc_format = {"doc_key": doc_key, "dataset": "bacteria", "sentences": [], "tok_start_idx": [],
                     "entity_start_id": entity_start_id, "ner": [], "ner_num": [], "nested_entity": [],
                     "relations": [], "rel_num": []}

    for sent in processed_doc:
        # Get the tokens.
        toks = [tok.text for tok in sent]
        start_indices = [tok.idx for tok in sent]
        # end_indices = [tok.idx + len(tok.text) for tok in sent]
        # Align entities.
        entities_sent = get_entities_in_sent(sent, entities)
        aligned, continuous, missed = align_entities(sent, entities_sent)
        # for one_align in aligned.values():
        #     if len(one_align) != 3:
        #         print(one_align)
        #     if len(one_align[0]) == 1:
        #         count_continuous += 1
        #     elif len(one_align[0]) == 2:
        #         entity_len_2 += 1
        #     else:
        #         entity_len_more += 1
        #         print("length of entity more than 3: ", len(one_align[0]))
        # Align relations.
        relations_sent, keys_found, rel_missed, rel_discon_ent = get_relations_in_sent(aligned, continuous, relations)

        # Append to result list
        scierc_format["sentences"].append(toks)
        scierc_format["tok_start_idx"].append(start_indices)
        # scierc_format["tok_end_idx"].append(end_indices)
        entities_to_scierc = [list(x) for x in continuous.values()]
        nested_entity = is_nested_entities(entities_to_scierc)
        scierc_format["ner"].append(entities_to_scierc)
        scierc_format["nested_entity"].append(nested_entity)
        ner_num = len(aligned) + len(missed)
        scierc_format["ner_num"].append(ner_num)
        scierc_format["relations"].append(relations_sent)
        rel_num = len(relations_sent) + len(rel_missed)
        scierc_format["rel_num"].append(rel_num)

        # Keep track of which entities and relations we've found and which we haven't.
        entities_seen |= set(entities_sent["entity_id"])
        entities_alignment |= set(aligned.keys())
        entities_continuous |= set(continuous.keys())
        entities_no_alignment |= set(missed.keys())
        relations_found |= keys_found
        rel_missed_sum += len(rel_missed)
        rel_discon_ent_sum += len(rel_discon_ent)
        nested_entity_sum += sum(nested_entity)
    # Update counts.
    entities_missed = set(entities["entity_id"]) - entities_seen
    if len(entities_missed) > 0:
        print("entity_missed: ", entities_missed)
    relations_missed = set(relations.keys()) - relations_found
    # add cross_relation in scierc_format
    if len(relations_missed) > 0 and len(relations_missed) != rel_missed_sum:
        print("relation_missed: ", relations_missed, " rel_missed_sum: ", rel_missed_sum)

    COUNTS["entities_correct"] += len(entities_alignment)
    COUNTS["entities_continuous"] += len(entities_continuous)
    COUNTS["entities_nested"] += nested_entity_sum
    COUNTS["entities_misaligned"] += len(entities_no_alignment)
    COUNTS["entities_missed"] += len(entities_missed)
    COUNTS["entities_total"] += len(entities)
    COUNTS["relations_found"] += len(relations_found)
    COUNTS["relations_missed"] += rel_missed_sum  # len(relations_missed)
    COUNTS["relations_discon_ent"] += rel_discon_ent_sum
    COUNTS['relations_total'] += len(relations)
    # COUNTS['continuous_entities_total'] += count_continuous
    # COUNTS['entity_len_2_total'] += entity_len_2
    # COUNTS['entity_len_more_total'] += entity_len_more
    return scierc_format

def one_fold(fold):
    directory = "data/bb"
    print(f"Processing fold {fold}.")
    raw_subdirectory = "/processed_data/merge_data"
    df_abstracts = pd.read_table(f"{directory}/{raw_subdirectory}/BB_{fold}/bb_{fold}_abstracts.tsv",
                                 header=None, keep_default_na=False,
                                 names=["doc_key", "title", "abstract"])
    # one_doc = df_abstracts.query("doc_key=='F-22177851-007'")
    # print(one_doc['title'])
    # print(one_doc['abstract'])
    # char_start_end_id may contain two spans (discontiguous entity) divided by ";", check if it can get all span ids
    df_entities = pd.read_table(f"{directory}/{raw_subdirectory}/BB_{fold}/bb_{fold}_entities.tsv",
                                header=None, keep_default_na=False,
                                names=["doc_key", "entity_id", "label", "char_start_end_ids", "text"])
    # e_t is expressed as "Entity Type:Entity_id", we need to judge entity type to get a correct relation
    df_relations = pd.read_table(f"{directory}/{raw_subdirectory}/BB_{fold}/bb_{fold}_relations.tsv",
                                 header=None, keep_default_na=False,
                                 names=["doc_key", "rel_id", "label", "e_t1", "e_t2"])
    df_entity_start_ids = pd.read_table(f"{directory}/{raw_subdirectory}/BB_{fold}/bb_{fold}_entity_start_ids.tsv",
                                        header=None, keep_default_na=False,
                                        names=["doc_key", "entity_start_id"])

    res_train = []
    res_eval = []
    for _, abstract in tqdm(df_abstracts.iterrows(), total=len(df_abstracts)):
        to_append = one_abstract(abstract, df_entities, df_relations, df_entity_start_ids)
        to_append_train = {"doc_key": to_append["doc_key"],
                           "dataset": to_append["dataset"],
                           "sentences": to_append["sentences"],
                           "ner": to_append["ner"],
                           "relations": to_append["relations"]}
        res_train.append(to_append_train)
        res_eval.append(to_append)

    # Write to file for training.
    name_out = f"{directory}/processed_data/json/{fold}.jsonl"
    if not os.path.exists(f"{directory}/processed_data/json"):
        os.makedirs(f"{directory}/processed_data/json")

    with open(name_out, "w") as f_out:
        for line in res_train:
            print(json.dumps(line), file=f_out)

    # Write to file for evaluation.
    name_out = f"{directory}/processed_data/json/eval_{fold}.jsonl"
    with open(name_out, "w") as f_out:
        for line in res_eval:
            print(json.dumps(line), file=f_out)


####################

# Driver

COUNTS = Counter()

for fold in ["train", "dev"]:   # , "dev"
    one_fold(fold)

counts = pd.Series(COUNTS)
print()
print("Some entities were missed due to tokenization choices in SciSpacy. Here are the stats:")
print(counts)
