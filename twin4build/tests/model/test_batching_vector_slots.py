"""Batching keeps every instance's vector slot.

A hub with vector ports (an air handler with one slot per branch) reads one
scalar from each of many leaves and hands its slots to many sinks, one of
which reads two slots through one connection (a room with two terminals).
Batched, the leaves are one meta component and so are the sinks, and the
hub's slots must still map to the right instance: batched and unbatched
simulations agree in object mode and in functional mode.  Before, the
batcher kept one connection per (meta, port) pair with the first
instance's slots only.
"""

import datetime
import unittest

import torch
from dateutil import tz

import twin4build as tb
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter, State


def _n_t(start_time, end_time, step_size):
    _, _, n_t, _ = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
    return n_t


class Leaf(core.System):
    """``v = p`` (a parameter per instance)."""

    v = Output()
    p = Parameter(1.0, lb=0.0, ub=100.0)

    def forward(self, x, inputs, p, step_size):
        return x, {"v": p["p"].reshape(-1)}


def _leaf(p, p_max, id):
    """A leaf with its own cap in its ``parameter`` spec (a room's occupancy
    bound from its floor area, say) while the declared bounds are the class
    defaults: the batched meta must report the spec's caps per instance."""
    leaf = Leaf(p=p, id=id)
    leaf.parameter["p"] = {"lb": 0.0, "ub": float(p_max)}
    return leaf


class Hub(core.System):
    """``y[k] = 2 x[k]`` per slot; the width comes from the wiring."""

    x = Input(size="wired")
    y = Output(size=x)

    def forward(self, x, inputs, p, step_size):
        return x, {"y": 2.0 * inputs["x"]}


class Sink(core.System):
    """``acc <- acc + 3 sum_k u[k]`` over its slots (a zone integrating its
    branches); ``w`` is the new accumulator.  Stateful, so the functional
    engine has something to roll."""

    u = Input(size="wired")
    w = Output()
    acc = State(init=0.0)

    def forward(self, x, inputs, p, step_size):
        x_next = x + 3.0 * inputs["u"].sum(dim=-1, keepdim=True)
        return x_next, {"w": x_next[..., 0]}


N_LEAVES = 4
START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)
END = START + datetime.timedelta(minutes=30)
STEP = 600


def build():
    """Leaves 0..N feed hub slots 0..N; sink 0 reads slots 0 and N through one
    connection (two terminals), sinks 1..N-1 read slot k."""
    model = tb.Model(id="batching_vector_slots")
    hub = Hub(id="hub")
    leaves = [_leaf(p=float(i + 1), p_max=10.0 * (i + 1), id=f"leaf{i}") for i in range(N_LEAVES + 1)]
    sinks = [Sink(id=f"sink{k}") for k in range(N_LEAVES)]
    for i, leaf in enumerate(leaves):
        model.add_connection(leaf, hub, "v", "x", input_port_index=i)
    model.add_connection(
        hub, sinks[0], "y", "u",
        output_port_index=torch.tensor([0, N_LEAVES]), input_port_index=torch.tensor([0, 1]),
    )
    for k in range(1, N_LEAVES):
        model.add_connection(hub, sinks[k], "y", "u", output_port_index=k, input_port_index=0)
    model.load(draw_semantic_model=False, draw_simulation_model=False)
    return model, sinks


def per_step():
    """Slot i carries leaf i's p = i + 1, doubled by the hub, tripled and summed by the sink."""
    out = {"sink0": 6.0 * (1 + (N_LEAVES + 1))}
    for k in range(1, N_LEAVES):
        out[f"sink{k}"] = 6.0 * (k + 1)
    return out


def sink_histories(model, batched=None):
    out = {}
    for k in range(N_LEAVES):
        cid = f"sink{k}"
        if batched is None:
            hist = model.components[cid].output["w"].history()
            out[cid] = hist.reshape(hist.shape[0], -1)[:, 0].detach().cpu().clone()
        else:
            meta, i_c = model._component_to_meta[cid]
            hist = meta.output["w"].history()
            out[cid] = hist.reshape(hist.shape[0], -1)[:, i_c].detach().cpu().clone()
    return out


class TestBatchingVectorSlots(unittest.TestCase):
    def test_batched_model_keeps_each_instances_slot(self):
        model, _ = build()
        sim = tb.Simulator(model)
        sim.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)
        reference = sink_histories(model)
        n_t = len(next(iter(reference.values())))
        for cid, value in per_step().items():
            self.assertAlmostEqual(float(reference[cid][-1]), n_t * value, msg=cid)

        batched = model.batch_components()
        # The single-slot sinks share one meta (the two-slot sink has its
        # own width, hence its own signature); every leaf shares one.
        self.assertEqual(len({id(model._component_to_meta[f"sink{k}"][0]) for k in range(1, N_LEAVES)}), 1)
        self.assertEqual(len({id(model._component_to_meta[f"leaf{i}"][0]) for i in range(N_LEAVES + 1)}), 1)
        batched.load(draw_semantic_model=False, draw_simulation_model=False)
        # The leaf meta reports its parameter once, one start value per instance.
        leaf_meta = model._component_to_meta["leaf0"][0]
        (entry,) = leaf_meta.get_estimable_parameters()
        self.assertEqual(entry[1], "p")
        self.assertEqual(list(entry[2]), [1.0, 2.0, 3.0, 4.0, 5.0])
        # ... and one upper bound per instance, not the first instance's
        # for all (a start above the first room's cap was rejected before);
        # a bound the instances share stays scalar.
        self.assertEqual(entry[3], 0.0)
        self.assertEqual(list(entry[4]), [10.0, 20.0, 30.0, 40.0, 50.0])

        for execution_mode in ("object", "functional"):
            with self.subTest(execution_mode=execution_mode):
                sim = tb.Simulator(batched, execution_mode=execution_mode, execution_backend="eager")
                sim.simulate(start_time=START, end_time=END, step_size=STEP, show_progress_bar=False)
                got = sink_histories(model, batched)
                for cid in reference:
                    torch.testing.assert_close(got[cid], reference[cid], msg=cid)
        # Written back, the original sinks carry their own instance's history
        # (a copied singleton shares its ports with its copy already).
        for k in range(1, N_LEAVES):
            model.components[f"sink{k}"].output["w"]._history.zero_()
        model.unbatch_histories(batched)
        for cid, hist in sink_histories(model).items():
            torch.testing.assert_close(hist, reference[cid], msg=cid)


if __name__ == "__main__":
    unittest.main()
