import numpy as np

from smartmarl.baselines.rule_based import GridTopology, fixed_time_actions, maxpressure_actions


def test_fixed_time_actions_shape_and_bounds():
    actions = fixed_time_actions(current_step=10, num_intersections=25, num_phases=4, cycle_length=60)
    assert actions.shape == (25,)
    assert np.all(actions >= 0)
    assert np.all(actions < 4)


def test_maxpressure_actions_shape_and_bounds():
    topo = GridTopology(grid_size=5, num_intersections=25)
    obs = {
        "queue_per_intersection": np.linspace(1.0, 10.0, 25, dtype=np.float32),
    }
    actions = maxpressure_actions(obs=obs, current_step=0, topology=topo, num_phases=4)
    assert actions.shape == (25,)
    assert np.all(actions >= 0)
    assert np.all(actions < 4)
