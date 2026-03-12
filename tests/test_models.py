import torch

from smartmarl.models.actor import GATv2Actor
from smartmarl.models.hetgnn import HetGNN


def test_hetgnn_relation_weights_not_shared():
    model = HetGNN(hidden_dim=128, num_layers=3)

    node_features = {
        "int": torch.randn(25, 3),
        "lane": torch.randn(25, 4),
        "sens": torch.randn(25, 4),
        "inj": torch.randn(25, 4),
    }
    edge_index = {
        "spatial": torch.randint(0, 25, (2, 100)),
        "flow_lane": torch.stack([torch.arange(25), torch.arange(25)]),
        "flow_sens": torch.stack([torch.arange(25), torch.arange(25)]),
        "incident": torch.stack([torch.arange(25), torch.arange(25)]),
    }
    _ = model(node_features, edge_index)

    w = model.relation_weight_tensors()
    assert w["spatial"].data_ptr() != w["flow"].data_ptr()
    assert w["flow"].data_ptr() != w["incident"].data_ptr()
    assert w["spatial"].data_ptr() != w["incident"].data_ptr()


def test_gatv2_attention_is_dynamic_for_same_structure():
    actor = GATv2Actor(input_dim=128, num_heads=2, head_dim=64, num_phases=4)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]],
        dtype=torch.long,
    )

    h1 = torch.randn(4, 128)
    h2 = h1.clone()
    h2[0] += 5.0

    _, attn1 = actor(h1, edge_index, return_attention=True)
    _, attn2 = actor(h2, edge_index, return_attention=True)

    diff = torch.mean(torch.abs(attn1[0] - attn2[0])).item()
    assert diff > 1e-4, "Attention coefficients did not change across feature snapshots"
