import importlib
import importlib.util
import json

import torch
from torch_geometric.data import Data


PROVIDERS = [
    "alexandria_pbe",
    "alexandria_pbesol",
    "matterverse",
    "mp",
    "mpdd",
    "nmd",
    "oqmd",
    "twodmatpedia",
]


def source_graph(provider, split, value):
    provider_targets = {
        f"{provider}.band_gap.ev": value,
        f"{provider}.formation_energy.ev_per_atom": value + 1,
        f"{provider}.hull_distance.ev_per_atom": value + 2,
        f"{provider}.scan_band_gap.ev": value + 3,
        f"{provider}.scan_hull_distance.ev_per_atom": value + 4,
        f"{provider}.energy_above_hull.ev_per_atom": value,
        f"{provider}.formation_energy_sipfenn_light.ev_per_atom": value,
        f"{provider}.formation_energy_sipfenn_novel.ev_per_atom": value,
        f"{provider}.formation_energy_sipfenn_standard.ev_per_atom": value,
        f"{provider}.stability_sipfenn.ev_per_atom": value,
        f"{provider}.delta_e.ev_per_atom": value,
        f"{provider}.stability.ev_per_atom": value,
    }
    common_targets = {
        "common.band_gap.ev": value,
        "common.thermo_stability.ev_per_atom": value + 1,
    }
    return Data(
        x=torch.randn(3, 12),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        edge_attr=torch.randn(3, 24),
        y=torch.tensor([0.0]),
        provider_id=provider,
        structure_id=f"{provider}-{value}",
        provider_targets=provider_targets,
        common_targets=common_targets,
        task_names=list(provider_targets),
        common_task_names=list(common_targets),
        split=split,
    )


def test_builder_creates_uniform_eight_client_masked_release(tmp_path):
    assert importlib.util.find_spec("scripts.prepare_data.build_optimade_release") is not None
    builder = importlib.import_module("scripts.prepare_data.build_optimade_release")
    source = tmp_path / "source"
    output = tmp_path / "output"
    for client_id, provider in enumerate(PROVIDERS):
        shard_dir = source / "clients" / provider / "shards"
        shard_dir.mkdir(parents=True)
        torch.save(
            [source_graph(provider, "train", float(client_id)), source_graph(provider, "test", float(client_id + 1))],
            shard_dir / "shard_000000.pt",
        )

    report = builder.build_release(source, output)

    assert report["num_clients"] == 8
    assert report["num_targets"] == 19
    manifest = json.loads((output / "fedgb_manifest.json").read_text())
    assert manifest["num_clients"] == 8
    assert manifest["feature_dim"] == 12
    assert manifest["edge_feature_dim"] == 24
    partition = output / "distrib" / manifest["processed_partition"]
    global_ids = []
    expected_task_counts = [5, 7, 0, 3, 4, 0, 5, 0]
    for client_id, provider in enumerate(PROVIDERS):
        payload = torch.load(partition / f"data_{client_id}.pt", weights_only=False)
        assert payload.client_name == provider
        assert payload.num_targets == 19
        assert len(payload.target_names) == 19
        assert len(payload.active_target_names) == expected_task_counts[client_id]
        assert payload.y.shape == (2, 19)
        assert payload.y_mask.shape == (2, 19)
        assert int(payload.y_mask[0].sum()) == expected_task_counts[client_id]
        assert not hasattr(payload.graphs[0], "provider_targets")
        assert not hasattr(payload.graphs[0], "common_targets")
        assert payload.graphs[0].fgl_client_id == client_id
        global_ids.extend(payload.global_map.values())
        split = partition / "graph_reg" / "default_split"
        assert torch.load(split / f"train_{client_id}.pt", weights_only=False).tolist() == [True, False]
        assert torch.load(split / f"test_{client_id}.pt", weights_only=False).tolist() == [False, True]
    assert global_ids == list(range(16))
