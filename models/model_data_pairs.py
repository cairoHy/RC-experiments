# make sure the model supports the dataset you use
models_in_datasets = {
    "CBT_NE": ["AttentionSumReader", "AoAReader"],
    "CBT_CN": ["AttentionSumReader", "AoAReader"],
    "SQuAD": ["RNet"]
}
