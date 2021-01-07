local template = import "template.libsonnet";

template.SpanMB {
  bert_model: "bert/BiomedNLP-PubMedBERT-base-uncased-abstract",
  cuda_device: 2,
  data_paths: {
    train: "data/chemprot/collated_data/standard/training.jsonl",
    validation: "data/chemprot/collated_data/standard/development.jsonl",
    test: "data/chemprot/processed_data/standard/test.jsonl",
  },
  loss_weights: {
    ner: 0.2,
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
  },

}
