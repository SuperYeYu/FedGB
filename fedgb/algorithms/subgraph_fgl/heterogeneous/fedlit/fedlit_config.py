config = {
    "supported_datasets": ["MUTAG", "pubmed_diabetes"],
    "fedlit_task": "classification",
    "fedlit_partition": "balanced",
    "fedlit_test_linktypes": "0-1-2-3",
    "fedlit_nfeature": 200,
    "fedlit_nclass": 3,
    "fedlit_nlinktype": 4,
    "fedlit_num_iter_em": 1,
    "fedlit_edge_batchsize": 2000000,
    "fedlit_local_epoch": 1,
    "fedlit_hidden_dim": 64,
    "fedlit_dropout": 0.3,
}
