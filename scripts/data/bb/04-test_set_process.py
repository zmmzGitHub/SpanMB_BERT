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

def re_sent_token(doc):
    # re-sentence: merge and sep
    new_sents_list = []
    sent_number = len(list(doc.sents))
    # print("Before sents number: ", sent_number)
    dump_sent = None
    for sent in doc.sents:
        print("add_id: ", add_id)
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
            if sent.text.startswith('hpaJ'):
                new_sents_list.append(dump_sent)
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
    print("sent1: ", sent1.text, "\nsent2: ", sent2.text)
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

        print("tok: ", tok.text, ' tok split: ', toks)
        add_id[0] += len(toks) - 1

            # break  # only once
    if len(new_toks) == 0:
        new_toks.append(NToken(tok.idx, tok.i + add_id[0], tok.text))

    return new_toks

####################

# Manage a single document and a single fold.

def one_abstract(row, df_entity_start_ids):
    doc_key = row["doc_key"]
    # print(type(row), type(row['doc_key']))
    if 'F' in doc_key:
        print(doc_key + ' is a para')
        doc = row["abstract"]
    else:
        print(doc_key + ' is a abstract')
        doc = row["title"] + " " + row["abstract"]

    entity_start_id = df_entity_start_ids.query(f"doc_key == '{doc_key}'").iat[0, 1]

    processed = nlp(doc)

    processed_doc = re_sent_token(processed)

    scierc_format = {"doc_key": doc_key, "dataset": "bacteria", "sentences": [],
                     "tok_start_idx": [], "entity_start_id": entity_start_id}

    for sent in processed_doc:
        # Get the tokens.
        toks = [tok.text for tok in sent]
        start_indices = [tok.idx for tok in sent]
        # end_indices = [tok.idx + len(tok.text) for tok in sent]

        # Append to result list
        scierc_format["sentences"].append(toks)
        scierc_format["tok_start_idx"].append(start_indices)
        # scierc_format["tok_end_idx"].append(end_indices)

    return scierc_format

def one_fold(fold):
    directory = "data/bacteria"
    print(f"Processing fold {fold}.")
    raw_subdirectory = "/processed_data/merge_data"
    df_abstracts = pd.read_table(f"{directory}/{raw_subdirectory}/BB_{fold}/bb_{fold}_abstracts.tsv",
                                 header=None, keep_default_na=False,
                                 names=["doc_key", "title", "abstract"])
    df_entity_start_ids = pd.read_table(f"{directory}/{raw_subdirectory}/BB_{fold}/bb_{fold}_entity_start_ids.tsv",
                                        header=None, keep_default_na=False,
                                       names=["doc_key", "entity_start_id"])

    res = []
    for _, abstract in tqdm(df_abstracts.iterrows(), total=len(df_abstracts)):
        to_append = one_abstract(abstract, df_entity_start_ids)
        res.append(to_append)

    # Write to file.
    name_out = f"{directory}/processed_data/json/eval_{fold}.jsonl"
    if not os.path.exists(f"{directory}/processed_data/json"):
        os.makedirs(f"{directory}/processed_data/json")

    with open(name_out, "w") as f_out:
        for line in res:
            print(json.dumps(line), file=f_out)


####################

# Driver
for fold in ["test"]:
    one_fold(fold)

