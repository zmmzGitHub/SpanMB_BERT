import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm
import json
import re, os
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
nlp = spacy.load("en_core_sci_sm")     # tokenizer, "en_core_sci_md" is bigger
add_id = [0]

####################

# Process entities and relations for a given abstract.

def get_entities_in_sent(sent, entities):
    start, end = sent.start_char, sent.end_char
    start_ok = entities["char_start"] >= start
    end_ok = entities["char_end"] <= end
    keep = start_ok & end_ok
    res = entities[keep]
    return res


def align_one(sent, row):
    # Don't distinguish b/w genes that can and can't be looked up in database.
    lookup = {"GENE-Y": "GENE",
              "GENE-N": "GENE",
              "CHEMICAL": "CHEMICAL"}

    start_tok = None
    end_tok = None

    for tok in sent:
        if tok.idx == row["char_start"]:
            start_tok = tok
        if tok.idx + len(tok) == row["char_end"]:
            end_tok = tok

    if start_tok is None or end_tok is None:
        # print("sent: ", sent.text, " entity: ", row.text)
        # print("en_start: ", row["char_start"], " en_end: ", row["char_end"])
        # print("sent toks: ", [(tok.idx, tok.text) for tok in sent])
        return None
    else:
        expected = sent.text[start_tok.idx - sent.start_char:end_tok.idx - sent.start_char + len(end_tok)]
        if expected != row.text:
            # print("expected: ", expected, " row tex: ", row.text)
            # print("ent_start: ", row["char_start"], " ent_end: ", row["char_end"])
            # print("sent toks: ", [(tok.idx, tok.text) for tok in sent])
            raise Exception("Entity mismatch")

        return (start_tok.i, end_tok.i, lookup[row["label"]])


def align_entities(sent, entities_sent):
    aligned_entities = {}
    missed_entities = {}
    for _, row in entities_sent.iterrows():
        aligned = align_one(sent, row)
        if aligned is not None:
            aligned_entities[row["entity_id"]] = aligned
        else:
            missed_entities[row["entity_id"]] = None

    return aligned_entities, missed_entities


def format_relations(relations):
    # Convert to dict.
    res = {}
    all_rels = []   # one entity-pair may has multiple relation types
    for _, row in relations.iterrows():
        ent1 = row["arg1"].replace("Arg1:", "")
        ent2 = row["arg2"].replace("Arg2:", "")
        key = (ent1, ent2)
        res[key] = row["label"]  # actually, all relation type are divided into ten groups, i.e. cpr_group

    return res


def get_relations_in_sent(aligned, relations):
    res = []
    keys = set()
    # Loop over the relations, and keep the ones relating entities in this sentences.
    for ents, label in relations.items():
        if ents[0] in aligned and ents[1] in aligned:
            keys.add(ents)
            ent1 = aligned[ents[0]]
            ent2 = aligned[ents[1]]
            to_append = ent1[:2] + ent2[:2] + (label,)
            res.append(to_append)

    return res, keys

def re_sent_token(doc):
    # all_entity_ids = set()
    # for _, entity in entities.iterrows():
    #     ids = re.split('[\s;]', entity["char_start_end_ids"])
    #     all_entity_ids |= set(ids)

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
                    dump_sent = re_sent(sent)
                else:
                    dump_sent = merge_sent(dump_sent, sent)
            else:
                dump_sent = re_sent(sent)
        else:
            if dump_sent is None:
                dump_sent = re_sent(sent)
            else:
                dump_sent = merge_sent(dump_sent, sent)
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

def merge_sent(sent1, sent2):
    # print("sent1: ", sent1.text, "\nsent2: ", sent2.text)
    # print("sent1 id: ", sent1.end_char, " sent2 id:", sent2.start_char)
    diff_id = sent2.start_char-sent1.end_char
    new_toks = []
    for tok in sent1:     # sent1 is dump_sent
        new_toks.append(tok)
    for tok in sent2:
        new_toks.extend(re_token(tok))
    new_sent = NSpan(sent1.start_char, sent2.end_char, new_toks, sent1.text + ' ' * diff_id + sent2.text)
    return new_sent

def re_sent(sent):
    new_toks = []
    for tok in sent:
        new_toks.extend(re_token(tok))

    new_sent = NSpan(sent.start_char, sent.end_char, new_toks, sent.text)
    return new_sent

def re_token(tok):
    new_toks = []

    toks = None
    # Chemprot text contains many '(' and ')'. To limit the number of tokens in one sentence,
    # we do not split '(' and ')' in token
    if re.search('([-/])', tok.text):
        toks = re.split('([-/])', tok.text)
    # elif re.search('(\d+)', tok.text):
    #     toks = re.split('(\d+)', tok.text)  # one special tok
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

def one_abstract(row, df_entities, df_relations):
    doc = row["title"] + " " + row["abstract"]
    doc_key = row["doc_key"]
    # print("doc_key: ", doc_key)
    entities = df_entities.query(f"doc_key == '{doc_key}'")
    relations = format_relations(df_relations.query(f"doc_key == '{doc_key}'"))

    processed = nlp(doc)
    processed_doc = re_sent_token(processed)

    entities_seen = set()
    entities_alignment = set()
    entities_no_alignment = set()
    relations_found = set()

    scierc_format = {"doc_key": doc_key, "dataset": "chemprot", "sentences": [], "ner": [],
                     "ner_num": [], "relations": [], "rel_num": []}

    for sent in processed_doc:
        # Get the tokens.
        toks = [tok.text for tok in sent]

        # Align entities.
        entities_sent = get_entities_in_sent(sent, entities)
        aligned, missed = align_entities(sent, entities_sent)

        # Align relations.
        relations_sent, keys_found = get_relations_in_sent(aligned, relations)

        # Append to result list
        scierc_format["sentences"].append(toks)
        entities_to_scierc = [list(x) for x in aligned.values()]
        scierc_format["ner"].append(entities_to_scierc)
        scierc_format["relations"].append(relations_sent)

        # Keep track of which entities and relations we've found and which we haven't.
        entities_seen |= set(entities_sent["entity_id"])
        entities_alignment |= set(aligned.keys())
        entities_no_alignment |= set(missed.keys())
        relations_found |= keys_found

    # Update counts.
    entities_missed = set(entities["entity_id"]) - entities_seen
    relations_missed = set(relations.keys()) - relations_found

    COUNTS["entities_correct"] += len(entities_alignment)
    COUNTS["entities_misaligned"] += len(entities_no_alignment)
    COUNTS["entities_missed"] += len(entities_missed)
    COUNTS["entities_total"] += len(entities)
    COUNTS["relations_found"] += len(relations_found)
    COUNTS["relations_missed"] += len(relations_missed)
    COUNTS['relations_total'] += len(relations)

    return scierc_format


def one_fold(fold):
    directory = "data/chemprot"
    print(f"Processing fold {fold}.")
    raw_subdirectory = "raw_data/ChemProt_Corpus"
    df_abstracts = pd.read_table(f"{directory}/{raw_subdirectory}/chemprot_{fold}/chemprot_{fold}_abstracts.tsv",
                                 header=None, keep_default_na=False, quoting=3,      # keep quotechar
                                 names=["doc_key", "title", "abstract"])
    # one_doc = df_abstracts.query("doc_key=='23194825'")
    # print(one_doc['title'])
    # print(one_doc['abstract'])
    # char_start_end_id may contain two spans (discontiguous entity) divided by ";", check if it can get all span ids

    df_entities = pd.read_table(f"{directory}/{raw_subdirectory}/chemprot_{fold}/chemprot_{fold}_entities.tsv",
                                header=None, keep_default_na=False,
                                names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"])
    df_relations = pd.read_table(f"{directory}/{raw_subdirectory}/chemprot_{fold}/chemprot_{fold}_relations.tsv",
                                 header=None, keep_default_na=False,
                                 names=["doc_key", "cpr_group", "eval_type", "label", "arg1", "arg2"])

    res = []
    for _, abstract in tqdm(df_abstracts.iterrows(), total=len(df_abstracts)):
        to_append = one_abstract(abstract, df_entities, df_relations)
        res.append(to_append)

    # Write to file.
    name_out = f"{directory}/processed_data/{fold}.jsonl"
    if not os.path.exists(f"{directory}/processed_data"):
        os.makedirs(f"{directory}/processed_data")
    with open(name_out, "w") as f_out:
        for line in res:
            print(json.dumps(line), file=f_out)


####################

# Driver

COUNTS = Counter()

for fold in ["development"]:    # "training", "development", "test"
    one_fold(fold)

counts = pd.Series(COUNTS)
print()
print("Some entities were missed due to tokenization choices in SciSpacy. Here are the stats:")
print(counts)
