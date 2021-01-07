local template = import "template.libsonnet";

template.SpanMB {
  bert_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
  cuda_device: 0,
  data_paths: {
    train: "data/bb/collated_data/train.jsonl",
    validation: "data/bb/collated_data/dev.jsonl",
    test: "data/bb/collated_data/dev.jsonl",
  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0
  },
  target_task: "relation",
  model +: {
    feedforward_params +: {
      num_layers: 1
    },
  },
  trainer +: {
    num_epochs: 25
  }
}
