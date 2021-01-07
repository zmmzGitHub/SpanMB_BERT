import os
import pandas as pd

# 这里将Bacteria Biotope提供的分散数据集合并成跟chemprot类似的格式。
path = 'data/bb'

# # test path
# print(os.path.exists(path))
# one_file = f"{path}/t2.txt"
# print(one_file)
# with open(one_file, 'r') as f:
#     print(f.read())

def merge_one(fold):
    raw_subdir = f"{path}/BB-rel+ner/BioNLP-OST-2019_BB-rel+ner_{fold}"
    all_file = os.listdir(raw_subdir)
    text_cont = []
    entity_cont = []
    rel_cont = []
    entity_start_id_cont = []
    for f_sub in all_file:
        if '.' not in f_sub:
            continue
        doc_key = None
        if 'F' in f_sub:
            doc_key = f_sub[f_sub.index('F'):f_sub.index('.')]    # paragraph of one full-text
        else:
            doc_key = f_sub[f_sub.rindex('-')+1:f_sub.index('.')]  # title and abstract of one article
        one_abstr = []
        if f_sub.endswith('.txt'):  # abstract and para
            with open(f"{raw_subdir}/{f_sub}", 'r') as f:
                one_abstr.append(doc_key)
                one_ta = []
                if 'F' not in doc_key:         # title and abstract
                    one_abstr.append(f.readline().strip())   # title
                    for line in f.readlines():
                        line = line.strip()
                        if line is not '':
                            one_ta.append(line)
                        else:
                            break
                    one_abstr.append(' '.join(one_ta))
                else:                          # para
                    one_abstr.append('')      # empty title col
                    one_para = ''
                    for line in f.readlines():        # paragraph may have multiple lines
                        # line = line.strip()
                        line = line.replace('\n', ' ')    # to match gold entity index
                        # if line is not '':
                        #     one_ta.append(line)
                        if line is not ' ':
                            one_para += line
                        else:
                            break
                    one_abstr.append(one_para)
                    # one_abstr.append(' '.join(one_ta))
                text_cont.append(one_abstr)
                assert len(text_cont[-1]) == 3

        elif f_sub.endswith('.a2'):   # entity and relation  test does not have a '.a2'
            with open(f"{raw_subdir}/{f_sub}", 'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    one_entity = []
                    one_rel = []
                    if line.startswith('T'):  # entity
                        one_entity.append(doc_key)
                        cols = line.split('\t')
                        for i in range(len(cols)):
                            if i == 1:
                                secs = cols[i].split(' ', 1)
                                one_entity.append(secs[0])
                                one_entity.append(secs[1])
                            else:
                                one_entity.append(cols[i])
                        entity_cont.append(one_entity)
                    if line.startswith("R"):  # relation
                        one_rel.append(doc_key)
                        for col in line.split():
                            one_rel.append(col)
                        rel_cont.append(one_rel)
        elif f_sub.endswith('.a1'):
            one_entity_start_id = [doc_key]
            with open(f"{raw_subdir}/{f_sub}", 'r') as f:
                T_sum = 0
                for line in f.readlines():
                    line = line.rstrip()
                    if line.startswith('T'):  # entity
                        T_sum += 1
                one_entity_start_id.append(T_sum)
            entity_start_id_cont.append(one_entity_start_id)
        else:
            continue
    for cont in ['abstracts', 'entities', 'relations', 'entity_start_ids']:
        output_subdir = f"{path}/processed_data/merge_data/BB_{fold}/bb_{fold}_{cont}.tsv"
        if not os.path.exists(f"{path}/processed_data/merge_data/BB_{fold}"):
            os.makedirs(f"{path}/processed_data/merge_data/BB_{fold}")

        if cont is 'abstracts':
            with open(output_subdir, 'w') as f:
                # tsv_w = csv.writer(f, delimiter='\t')
                # tsv_w.writerows(abstract_cont)
                df = pd.DataFrame(text_cont)
        elif cont is 'entities':
            with open(output_subdir, 'w') as f:
                # tsv_w = csv.writer(f, delimiter='\t')
                # tsv_w.writerows(entity_cont)
                df = pd.DataFrame(entity_cont)
        elif cont is "relations":
            with open(output_subdir, 'w') as f:
                # tsv_w = csv.writer(f, delimiter='\t')
                # tsv_w.writerows(rel_cont)
                df = pd.DataFrame(rel_cont)
        else:
            with open(output_subdir, "w") as f:
                df = pd.DataFrame(entity_start_id_cont)
        df.to_csv(output_subdir, sep='\t', index=None, header=None)

for fold in ['train', 'dev', 'test']:     # test set does not have a label set, i.e. file '.a2'
    merge_one(fold)

print('done!')
