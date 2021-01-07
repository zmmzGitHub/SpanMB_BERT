# SpanMB

Implements the model for joint entity and relation extraction from biomedical text about bacteria biotope.

## Table of Contents
- [Dependencies](#dependencies)
- [Model training](#training-a-model)
- [Model evaluation](#evaluating-a-model)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)

## Dependencies

Clone this repository and navigate the the root of the repo on your system. Install python=3.7. Then execute:

```
pip install -r requirements.txt
```

This repository adapts code from [DyGIE++](https://github.com/dwadden/dygiepp).

## Training a model

We only show steps to train a model here. The explanation of training details please refer to [DyGIE++](https://github.com/dwadden/dygiepp).

### Bacteria Biotope (BB)

The [BB-rel+ner 2019](https://sites.google.com/view/bb-2019/dataset) corpus contains two relation types and four entity types.  

The steps are as follows:
- **Get the data**.
  - Run `bash ./scripts/data/get_bb.sh`. This will download the data and process it into the model input format.
    - NOTE: Our model can only recognize relations between continuous entities within one sentence. We lose discontinuous entities, relations related discontinuous entities and cross-sentence relations. The obtained percentage of entities and relations can be checked by run `python scripts/data/bb/03_spot_check.py`. 
  - Collate the data:
    ```
    mkdir -p data/bb/collated_data
    
    python scripts/data/shared/collate.py \
      data/bb/processed_data \
      data/bb/collated_data \
      --train_name=train \
      --dev_name=dev
      
- **Train the model**. Enter `bash scripts/train bb`.

### ChemProt

The [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) corpus contains entity and relation annotations for drug / protein interaction.

Then, follow these steps:

- **Get the data**.
  - Run `bash ./scripts/data/get_chemprot.sh`. This will download the data and process it into the DyGIE input format.
    - NOTE: Compared to [DyGIE++](https://github.com/dwadden/dygiepp), we improve the preprocessing results by only losing 2% of the relations and the named entities. Run `python scripts/data/chemprot/03_spot_check.py` for a quick check.
  - Collate the data:
    ```
    mkdir -p data/chemprot/collated_data

    python scripts/data/shared/collate.py \
      data/chemprot/processed_data \
      data/chemprot/collated_data \
      --train_name=training \
      --dev_name=development

- **Train the model**. Enter `bash scripts/train chemprot`.


## Evaluating a model and making predictions on existing datasets

[DyGIE++](https://github.com/dwadden/dygiepp) shows details about evaluating a model and making predictions on a dataset. 
For evaluating the pre-trained model of bacteria biotope, run:
```  shell
allennlp evaluate \
  models/bb/model.tar.gz \
  data/bb/processed_data/json/eval_dev.jsonl \
  --cuda-device 0 \
  --include-package spanmb \
  --output-file models/bb/metrics_dev.json
```  
The evaluation for development set of bacteria biotope counts all gold entities and relations to compute the recall 
value and F1-score. Thus, lost entities and relations during preprocessing are considered false negatives.
For predicting on the test set of bacteria biotope, run:
```  shell
allennlp predict \
  models/bb/model.tar.gz \
  data/bb/processed_data/json/eval_test.jsonl \
  --predictor spanmb \
  --include-package spanmb \
  --use-dataset-reader \
  --output-file models/bb/test_output.json
  --cuda-device 0 \
  --silent
``` 
