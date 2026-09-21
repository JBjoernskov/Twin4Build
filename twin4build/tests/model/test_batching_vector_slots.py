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
import twin4build.utils.types as tps


def _n_t(start_time, end_time, step_size):
    _, _, n_t, _ = core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
    return n_t


class Leaf(core.System):
    """``v = p`` (a parameter per instance)."""

    def __init__(self, p=1.0, p_max=100.0, **kwargs):
        super().__init__(**kwargs)
        self.input = {}
        self.output = {"v": tps.Scalar()}
        # Each instance has its own cap (a room's occupancy bound from its
        # floor area, say): the batched meta must report them per instance.
        self.p = tps.Parameter(torch.tensor(float(p)), min_value=0.0, max_value=float(p_max))
        self.parameter = {"p": {"lb": 0.0, "ub": float(p_max)}}
        self._config = {"parameters": ["p"]}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size):
        n_t = _n_t(start_time, end_time, step_size)
        self.p = self.p.expand_to_n_c(self.n_c)
        self.output["v"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)

    PARAM_NAMES = ("p",)

    def forward(self, x, inputs, params, sample_time):
        return x, {"v": params["p"].reshape(-1)}

    def do_step(self, second_time, date_time, step_size, step_index):
        self.output["v"]._set(self.p.get().reshape(1, -1), i_t=step_index)


class Hub(core.System):
    """``y[k] = 2 x[k]`` per slot; the width comes from the wiring."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"x": tps.Vector()}
        self.output = {"y": tps.Vector()}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size):
        n_t = _n_t(start_time, end_time, step_size)
        indices = [
            int(i)
            for cp in self.connects_at
            if cp.input_port == "x"
            for conn in cp.connects_system_through
            for i in torch.as_tensor(cp.input_port_index[conn]).reshape(-1).tolist()
        ]
        n_v = max(indices, default=-1) + 1
        self.input["x"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c, n_v=n_v)
        self.output["y"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c, n_v=n_v)

    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        return x, {"y": 2.0 * inputs["x"]}

    def do_step(self, second_time, date_time, step_size, step_index):
        self.output["y"]._set(2.0 * self.input["x"].get(), i_t=step_index)


class Sink(core.System):
    """``acc <- acc + 3 sum_k u[k]`` over its slots (a zone integrating its
    branches); ``w`` is the new accumulator.  Stateful, so the functional
    engine has something to roll."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = {"u": tps.Vector()}
        self.output = {"w": tps.Scalar()}
        self.acc = tps.State(n_v=1, init_value=0.0, names=[f"{self.id}.acc"])
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size):
        n_t = _n_t(start_time, end_time, step_size)
        indices = [
            int(i)
            for cp in self.connects_at
            if cp.input_port == "u"
            for conn in cp.connects_system_through
            for i in torch.as_tensor(cp.input_port_index[conn]).reshape(-1).tolist()
        ]
        n_v = max(indices, default=-1) + 1
        self.input["u"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c, n_v=n_v)
        self.output["w"].initialize(n_t=n_t, n_s=len(start_time), n_c=self.n_c)
        self.acc.initialize(n_s=len(start_time), n_c=self.n_c, n_v=1, force=True)

    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        x_next = x + 3.0 * inputs["u"].sum(dim=-1, keepdim=True)
        return x_next, {"w": x_next[..., 0]}

    def do_step(self, second_time, date_time, step_size, step_index):
        x_next, outs = self.forward(self.get_state(), {"u": self.input["u"].get()}, {}, step_size)
        self.set_state(x_next)
        self.output["w"]._set(outs["w"], i_t=step_index)


N_LEAVES = 4
START = datetime.datetime(2024, 1, 1, tzinfo=tz.UTC)
END = START + datetime.timedelta(minutes=30)
STEP = 600


def build():
    """Leaves 0..N feed hub slots 0..N; sink 0 reads slots 0 and N through one
    connection (two terminals), sinks 1..N-1 read slot k."""
    model = tb.Model(id="batching_vector_slots")
    hub = Hub(id="hub")
    leaves = [Leaf(p=float(i + 1), p_max=10.0 * (i + 1), id=f"leaf{i}") for i in range(N_LEAVES + 1)]
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
        # ... and one bound per instance, not the first instance's for all
        # (a start above the first room's cap was rejected before).
        self.assertEqual(list(entry[3]), [0.0] * 5)
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
