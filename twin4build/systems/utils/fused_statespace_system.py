r"""Compile-time fusion of connected linear state-space components.

Why fusion exists
-----------------

Components are co-simulated Gauss-Seidel style: each steps once per timestep
against the other's held (one-step-lagged) outputs.  For a cycle like
``BuildingSpace <-> WallSystem`` the lag is unavoidable by ordering, and
when the exchanged signal is a stiff algebraic function of a neighbour's state
(a heat flow with gain :math:`1/R`) the loop gain :math:`\Delta t/(R\,C)` can
exceed 1 and the pair diverges -- for ANY execution order, because the loop
gain is invariant to where the cycle is cut.

:class:`FusedStateSpaceSystem` removes the problem at the root: clusters of
connected linear/bilinear state-space components are assembled into ONE
monolithic block whose continuous matrices couple the member states exactly
(the internal connections are eliminated algebraically), discretized with a
single exact ZOH step.  There is no intra-cluster lag at all, so the coupled
dynamics are exact and unconditionally stable by construction -- while every
parameter stays on, and is only ever consumed by, its owning component.

The elimination
---------------

Each member decomposes into state-space *units* (leaf systems exposing
``_build_matrices`` and ``_ss_layout``; a composite like
``BuildingSpaceSystem`` contributes its ``thermal`` and ``mass``
submodels).  For unit :math:`i` with continuous matrices
:math:`(A_i, B_i, C_i, D_i, E_i, F_i)` and input vector :math:`u_i`, every
*internal* input (fed by another member through a fusable connection) is
substituted by the sender's output row:

.. math::

   u_i[k] = y_j[l] = C_j[l,:]\,x_j + D_j[l,:]\,u_j

recursively until only states and external inputs remain (the feedthrough
chain must be acyclic -- asserted).  Stacking the units gives

.. math::

   u_i = S^x_i\,x + S^u_i\,u_{ext} \qquad\Rightarrow\qquad
   A[i\text{-rows}] = [\,0\;A_i\;0\,] + B_i S^x_i,\quad
   B[i\text{-rows}] = B_i S^u_i

which for a zone-wall pair reproduces literally the monolithic RC matrix that
a hand-derived combined model would have.  Bilinear terms (``E``/``F``) are
required to act on *external* inputs only (asserted) and are re-indexed into
the joint input vector.

Execution semantics
-------------------

The fused component is a normal :class:`~twin4build.systems.saref4syst.system.System`
that appears in the execution order INSTEAD of its members (the members stay
in ``model.components`` for parameter targeting, port history and user
access):

* ``forward`` is the pure one-step map over the joint state -- the single
  source of truth, used by ``do_step``, the composed fast paths and
  collocation alike.
* ``do_step`` reads the members' input ports (assigned generically through
  the synthesized connection-point views), takes one joint step and writes
  every member output port, so sensors and downstream components observe the
  members exactly as before.
* State lives in the members' own ``ss_model`` states (the generic
  ``get_state``/``set_state`` walk reaches them through the registered
  member modules), so estimator state seeding and collocation state layouts
  see one stateful component with the concatenated member state.
* Parameters are routed by prefixed names (``"<member_key>.<unit>.<name>"``),
  mirroring the composite-component convention; non-estimated entries fall
  back to the owning member's ``tps.Parameter`` defaults.

Published diagnostic outputs with feedthrough (e.g. the wall's heat flows)
are evaluated at the end-of-step state with held external inputs; this can
differ slightly from the unfused object graph (which used the lagged
neighbour value), but only in the *reported* signal -- the dynamics are
exact.
"""

# Standard library imports
import datetime
from typing import List, Optional, Tuple

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.discrete_statespace_system import bilinear_onestep
from twin4build.utils.rgetattr import rgetattr


def _module_key(component_id: str) -> str:
    """Sanitize a component id into a valid nn.Module attribute name."""
    key = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in component_id)
    if not key or key[0].isdigit():
        key = "m_" + key
    return key


class _ConnectionPointView:
    """Receiver-side view of a member's connection point, re-keyed to the
    fused component's namespaced port name.

    Carries the REAL ``Connection`` objects and the real index dicts, so the
    simulator's generic input assignment and the composer's source resolution
    work on the fused component unchanged -- the write lands on the member's
    own port object (aliased in ``fused.input``)."""

    __slots__ = (
        "input_port", "connects_system_through", "input_port_index",
        "output_port_index", "output_component_index", "input_component_index",
        "connection_point_of",
    )

    def __init__(self, fused, namespaced_port, real_cp, connections):
        self.input_port = namespaced_port
        self.connects_system_through = list(connections)
        self.input_port_index = real_cp.input_port_index
        self.output_port_index = real_cp.output_port_index
        self.output_component_index = real_cp.output_component_index
        self.input_component_index = real_cp.input_component_index
        self.connection_point_of = fused


class _ConnectionView:
    """Sender-side view of a member's outgoing connection, re-keyed to the
    fused component's namespaced output port (used by the required-
    initialization mapping after cycle cutting)."""

    __slots__ = ("connects_system", "output_port", "connects_system_at")

    def __init__(self, fused, namespaced_port, real_conn):
        self.connects_system = fused
        self.output_port = namespaced_port
        self.connects_system_at = real_conn.connects_system_at


class FusedStateSpaceSystem(core.System, nn.Module):
    """One monolithic state-space block replacing a cluster of connected
    linear/bilinear state-space components in the execution order.

    Args:
        members: The cluster's components, in a deterministic order (this
            order defines the joint state layout).
        internal_arcs: The eliminated connections, as tuples
            ``(sender, output_port, receiver, input_port, slot)`` where
            ``slot`` is the vector-port slot index (0 for scalar ports).
        id: Component id of the fused block.
    """

    PARAM_NAMES: tuple = ()

    def __init__(self, members: List, internal_arcs: List[tuple], **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self._members = list(members)
        self._member_keys = {}
        for m in self._members:
            key = _module_key(m.id)
            assert not hasattr(self, key), f"duplicate member module key {key}"
            self.add_module(key, m)
            self._member_keys[m.id] = key
        self._internal_arcs = list(internal_arcs)

        # (receiver_id, input_port, slot) -> (sender_id, output_port)
        self._internal_recv = {
            (r.id, r_port, int(slot)): (s.id, s_port)
            for (s, s_port, r, r_port, slot) in self._internal_arcs
        }
        internal_recv_ports = {(rid, port) for (rid, port, _) in self._internal_recv}
        internal_conns = set()
        for (s, s_port, r, r_port, slot) in self._internal_arcs:
            for cp in r.connects_at:
                if cp.input_port != r_port:
                    continue
                for conn in cp.connects_system_through:
                    if conn.connects_system is s and conn.output_port == s_port:
                        internal_conns.add(id(conn))

        # Namespaced port dicts ALIASING the member port objects: a write
        # through the fused dict is a write to the member port.
        fused_input = {}
        fused_output = {}
        for m in self._members:
            for port, obj in m.input.items():
                if (m.id, port) in internal_recv_ports:
                    continue  # eliminated into the joint matrices
                fused_input[f"{m.id}.{port}"] = obj
            for port, obj in m.output.items():
                fused_output[f"{m.id}.{port}"] = obj
        self.input = fused_input
        self.output = fused_output

        # Synthesized graph views (external arcs only).
        self.connects_at = []
        self.connected_through = []
        for m in self._members:
            for cp in m.connects_at:
                ext = [
                    conn for conn in cp.connects_system_through
                    if id(conn) not in internal_conns
                ]
                if ext:
                    self.connects_at.append(
                        _ConnectionPointView(
                            self, f"{m.id}.{cp.input_port}", cp, ext
                        )
                    )
            for conn in m.connected_through:
                if id(conn) not in internal_conns:
                    self.connected_through.append(
                        _ConnectionView(self, f"{m.id}.{conn.output_port}", conn)
                    )

        self._config = {"parameters": []}
        self._fwd_mat_cache = None
        self._do_step_params = None
        self.INITIALIZED = False

    @property
    def config(self):
        return self._config

    @property
    def members(self) -> List:
        return list(self._members)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(
        self,
        start_time: datetime.datetime = None,
        end_time: datetime.datetime = None,
        step_size: int = None,
    ) -> None:
        """Initialize the members (their own topology discovery still works on
        the original, un-contracted connections), then build the joint layout:
        unit table, state offsets, external input columns and internal-arc
        substitution wiring."""
        for m in self._members:
            m.initialize(start_time, end_time, step_size)

        # -- unit table: (member, unit_prefix, unit, layout, param_prefix) --
        self._units = []
        for m in self._members:
            key = self._member_keys[m.id]
            for prefix, unit in m._ss_units():
                param_prefix = f"{key}.{prefix}" if prefix else key
                self._units.append(
                    {
                        "member": m,
                        "unit": unit,
                        "layout": unit._ss_layout(),
                        "param_prefix": param_prefix,
                    }
                )

        # -- state offsets from the default-parameter matrices ---------------
        offset = 0
        for entry in self._units:
            A = entry["unit"]._build_matrices()[0]
            entry["n_states"] = A.shape[-1]
            entry["x_offset"] = offset
            offset += entry["n_states"]
        self._n_joint_states = offset

        # -- input columns: internal (substituted) or external (joint u) -----
        # External columns are keyed by the namespaced port name; the same
        # port may feed several unit columns (composite submodels sharing a
        # port object).
        self._ext_names: List[str] = []
        ext_index = {}
        for entry in self._units:
            m = entry["member"]
            cols = []  # per unit column: ("ext", j) | ("int", sender_id, out_port)
            for port, width in entry["layout"]["u"]:
                for slot in range(width):
                    src = self._internal_recv.get((m.id, port, slot))
                    if src is not None:
                        cols.append(("int", src[0], src[1]))
                    else:
                        name = f"{m.id}.{port}"
                        assert name in self._input, (
                            f"fused input port {name} missing (vector port "
                            "partially internal?)"
                        )
                        if name not in ext_index:
                            ext_index[name] = len(self._ext_names)
                            self._ext_names.append(name)
                        cols.append(("ext", ext_index[name]))
            entry["cols"] = cols

        # -- output rows: every member output, namespaced --------------------
        # (unit index, row) per published output name.
        self._out_names: List[str] = []
        self._out_rows: List[Tuple[int, int]] = []
        for i, entry in enumerate(self._units):
            m = entry["member"]
            for port, row in entry["layout"]["y"].items():
                self._out_names.append(f"{m.id}.{port}")
                self._out_rows.append((i, row))

        # sender (member_id, output_port) -> (unit index, row)
        self._sender_rows = {}
        for i, entry in enumerate(self._units):
            m = entry["member"]
            for port, row in entry["layout"]["y"].items():
                self._sender_rows[(m.id, port)] = (i, row)
        for (s, s_port, _, _, _) in self._internal_arcs:
            assert (s.id, s_port) in self._sender_rows, (
                f"internal arc source {s.id}.{s_port} is not a state-space "
                "output of the cluster"
            )

        self._fwd_mat_cache = None
        self._do_step_params = None
        self.INITIALIZED = True

    # ------------------------------------------------------------------
    # Joint matrix assembly
    # ------------------------------------------------------------------

    def _unit_params(self, entry, params):
        """Physical-parameter dict for one unit: estimated entries from the
        prefixed ``params`` (``"<member_key>.<unit>.<name>"``), the rest from
        the unit's own ``tps.Parameter`` defaults."""
        unit = entry["unit"]
        prefix = entry["param_prefix"]
        out = {}
        for name in unit.PARAM_NAMES:
            key = f"{prefix}.{name}"
            out[name] = params[key] if key in params else rgetattr(unit, name).get()
        return out

    def _assemble(self, params):
        """Build the joint ``(A, B, C, D, E, F)`` from the unit matrices,
        eliminating the internal arcs exactly."""
        units = self._units
        mats = [u["unit"]._build_matrices(self._unit_params(u, params)) for u in units]

        n_c = max(m[0].shape[0] for m in mats)
        N = self._n_joint_states
        M = len(self._ext_names)
        dtype = torch.float64

        def _expand(t):
            return t if t.shape[0] == n_c else t.expand(n_c, *t.shape[1:])

        # Substitution: per unit, u_i = S_x[i] @ x + S_u[i] @ u_ext.
        # Row cache keyed (unit index, column); cycle-guarded (a cyclic
        # feedthrough chain would be a purely algebraic loop -- not fusable).
        row_cache = {}

        def input_row(i, k, _busy=frozenset()):
            hit = row_cache.get((i, k))
            if hit is not None:
                return hit
            assert (i, k) not in _busy, (
                "algebraic loop in fused feedthrough chain -- cluster is not "
                "fusable"
            )
            kind = units[i]["cols"][k]
            if kind[0] == "ext":
                sx = torch.zeros((n_c, 1, N), dtype=dtype)
                su = torch.zeros((n_c, 1, M), dtype=dtype)
                su[:, 0, kind[1]] = 1.0
            else:
                j, row = self._sender_rows[(kind[1], kind[2])]
                Aj, Bj, Cj, Dj, Ej, Fj = mats[j]
                oj, nj = units[j]["x_offset"], units[j]["n_states"]
                sx = torch.zeros((n_c, 1, N), dtype=dtype)
                sx[:, 0, oj:oj + nj] = _expand(Cj)[:, row, :]
                su = torch.zeros((n_c, 1, M), dtype=dtype)
                Dj_row = _expand(Dj)[:, row, :]  # (n_c, m_j)
                if bool((Dj_row.detach() != 0).any()):
                    for kk in range(Dj_row.shape[-1]):
                        col = Dj_row[:, kk]
                        if not bool((col.detach() != 0).any()):
                            continue
                        sub_sx, sub_su = input_row(j, kk, _busy | {(i, k)})
                        sx = sx + col.reshape(n_c, 1, 1) * sub_sx
                        su = su + col.reshape(n_c, 1, 1) * sub_su
            row_cache[(i, k)] = (sx, su)
            return sx, su

        A = torch.zeros((n_c, N, N), dtype=dtype)
        B = torch.zeros((n_c, N, M), dtype=dtype)
        E = torch.zeros((n_c, M, N, N), dtype=dtype)
        F = torch.zeros((n_c, M, N, M), dtype=dtype)

        S = []  # per unit: (S_x (n_c, m_i, N), S_u (n_c, m_i, M))
        for i, entry in enumerate(units):
            m_i = len(entry["cols"])
            if m_i:
                rows = [input_row(i, k) for k in range(m_i)]
                sx = torch.cat([r[0] for r in rows], dim=1)
                su = torch.cat([r[1] for r in rows], dim=1)
            else:
                sx = torch.zeros((n_c, 0, N), dtype=dtype)
                su = torch.zeros((n_c, 0, M), dtype=dtype)
            S.append((sx, su))

        for i, entry in enumerate(units):
            Ai, Bi, Ci, Di, Ei, Fi = (_expand(t) for t in mats[i])
            o, n = entry["x_offset"], entry["n_states"]
            sx, su = S[i]
            A[:, o:o + n, o:o + n] += Ai
            A[:, o:o + n, :] = A[:, o:o + n, :] + Bi @ sx
            B[:, o:o + n, :] = B[:, o:o + n, :] + Bi @ su

            # Bilinear terms must act on external inputs only: substituted
            # (state-valued) bilinear inputs would create quadratic state
            # terms that no LTI block can represent.
            for k, kind in enumerate(entry["cols"]):
                Ek = Ei[:, k]  # (n_c, n, n)
                Fk = Fi[:, k]  # (n_c, n, m_i)
                has_E = bool((Ek.detach() != 0).any())
                has_F = bool((Fk.detach() != 0).any())
                if not (has_E or has_F):
                    continue
                assert kind[0] == "ext", (
                    f"bilinear input column {k} of unit {i} is internal -- "
                    "cluster is not fusable"
                )
                j = kind[1]
                if has_E:
                    E[:, j, o:o + n, o:o + n] += Ek
                if has_F:
                    for k2, kind2 in enumerate(entry["cols"]):
                        col = Fk[:, :, k2]  # (n_c, n)
                        if not bool((col.detach() != 0).any()):
                            continue
                        assert kind2[0] == "ext", (
                            "bilinear F term references an internal input -- "
                            "cluster is not fusable"
                        )
                        F[:, j, o:o + n, kind2[1]] += col

        # Published outputs: y = (C_i + D_i S_x) x + (D_i S_u) u_ext rows.
        P = len(self._out_names)
        C = torch.zeros((n_c, P, N), dtype=dtype)
        D = torch.zeros((n_c, P, M), dtype=dtype)
        for p, (i, row) in enumerate(self._out_rows):
            _, _, Ci, Di, _, _ = mats[i]
            o, n = self._units[i]["x_offset"], self._units[i]["n_states"]
            sx, su = S[i]
            C[:, p, o:o + n] += _expand(Ci)[:, row, :]
            Di_row = _expand(Di)[:, row:row + 1, :]  # (n_c, 1, m_i)
            C[:, p, :] = C[:, p, :] + (Di_row @ sx)[:, 0, :]
            D[:, p, :] = D[:, p, :] + (Di_row @ su)[:, 0, :]

        return A, B, C, D, E, F

    # ------------------------------------------------------------------
    # forward / do_step
    # ------------------------------------------------------------------

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step of the fused cluster: ``(state, inputs, params) ->
        (new_state, outputs)``.

        ``inputs`` is keyed by the namespaced external port names
        (``"<member_id>.<port>"``); ``params`` by prefixed parameter paths
        (``"<member_key>.<unit>.<name>"``); outputs cover every member output,
        namespaced.  Matrices are cached per params-dict identity (theta-only
        work, done once per theta in a sequential rollout)."""
        cache = getattr(self, "_fwd_mat_cache", None)
        if cache is None or cache[0] is not params or cache[2] != sample_time:
            cache = (params, self._assemble(params), sample_time, {})
            self._fwd_mat_cache = cache
        A, B, C, D, E, F = cache[1]
        u = torch.stack([inputs[name] for name in self._ext_names], dim=-1)
        x_next, y = bilinear_onestep(
            A, B, C, D, E, F, x, u, sample_time, disc_cache=cache[3]
        )
        outputs = {name: y[..., p] for p, name in enumerate(self._out_names)}
        return x_next, outputs

    def _forward_params(self) -> dict:
        """Params dict for the ``do_step`` path: every unit parameter under its
        prefixed name, from the members' own ``tps.Parameter`` values.

        Built ONCE per :meth:`initialize` (i.e. once per simulation):
        parameters cannot change during a simulation, and the stable dict
        identity lets :meth:`forward` reuse the assembled joint matrices for
        the whole rollout.  Estimation writes parameters and re-simulates,
        which re-initializes and rebuilds -- with the fresh autograd graph."""
        params = self._do_step_params
        if params is None:
            params = {
                f"{entry['param_prefix']}.{name}": rgetattr(
                    entry["unit"], name
                ).get()
                for entry in self._units
                for name in entry["unit"].PARAM_NAMES
            }
            self._do_step_params = params
        return params

    def do_step(
        self,
        second_time=None,
        date_time=None,
        step_size=None,
        step_index: Optional[int] = None,
    ) -> None:
        """One joint step: read the (already assigned) external member input
        ports, advance the joint state, write every member output port.

        Thin port-I/O wrapper around :meth:`forward` (single source of truth).
        The members' ``do_step`` is never called; their internal input ports
        (eliminated arcs) are not updated -- the coupling happens inside the
        joint matrices."""
        inputs = {name: self._input[name].get() for name in self._ext_names}
        x = self.get_state()  # (n_s, n_c, N) via the members' own states
        x_next, outs = self.forward(
            x, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.set_state(x_next)
        for name, value in outs.items():
            self._output[name]._set(value, i_t=step_index)
