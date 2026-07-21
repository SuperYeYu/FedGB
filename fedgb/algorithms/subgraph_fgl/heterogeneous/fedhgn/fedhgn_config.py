config = {
    "supported_datasets": ["MUTAG", "AIFB", "BGS"],
    "fedhgn_task": "node_classification",
    "fedhgn_num_bases": 8,
    "fedhgn_num_layers": 2,
    "fedhgn_hidden_dim": 64,
    "fedhgn_dropout": 0.3,
    "fedhgn_batch_size": 128,
    "fedhgn_local_epoch": 1,
    "fedhgn_align_reg": 0.5,
    "fedhgn_ablation": None,
    "fedhgn_use_self_loop": False,
    "fedhgn_max_nodes": None,
}
