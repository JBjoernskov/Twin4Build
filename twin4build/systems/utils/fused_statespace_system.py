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
    SUPPORTS_TRANSFORM_MODE = True

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
            unit = entry["unit"]
            matrices = unit._build_matrices()
            A, _, C, D, E, F = matrices
            entry["n_states"] = A.shape[-1]
            entry["x_offset"] = offset
            offset += entry["n_states"]
            support = unit._ss_support()
            assert set(support) == {"D", "E", "F"}, (
                f"{type(unit).__name__}._ss_support() must define exactly D, E and F"
            )

            d_by_row = {}
            for row, col in support["D"]:
                assert 0 <= row < D.shape[-2] and 0 <= col < D.shape[-1], (
                    f"{type(unit).__name__} declares out-of-range D support "
                    f"({row}, {col}) for shape {tuple(D.shape[-2:])}"
                )
                d_by_row.setdefault(row, []).append(col)

            e_inputs = set()
            for input_col, state_row, state_col in support["E"]:
                assert (
                    0 <= input_col < E.shape[-3]
                    and 0 <= state_row < E.shape[-2]
                    and 0 <= state_col < E.shape[-1]
                ), (
                    f"{type(unit).__name__} declares out-of-range E support "
                    f"({input_col}, {state_row}, {state_col}) for shape "
                    f"{tuple(E.shape[-3:])}"
                )
                e_inputs.add(input_col)

            f_by_input = {}
            for input_col, state_row, input_col_2 in support["F"]:
                assert (
                    0 <= input_col < F.shape[-3]
                    and 0 <= state_row < F.shape[-2]
                    and 0 <= input_col_2 < F.shape[-1]
                ), (
                    f"{type(unit).__name__} declares out-of-range F support "
                    f"({input_col}, {state_row}, {input_col_2}) for shape "
                    f"{tuple(F.shape[-3:])}"
                )
                f_by_input.setdefault(input_col, set()).add(input_col_2)

            entry["support"] = {
                "D_by_row": {
                    row: tuple(sorted(cols)) for row, cols in d_by_row.items()
                },
                "E_inputs": frozenset(e_inputs),
                "F_by_input": {
                    col: tuple(sorted(cols)) for col, cols in f_by_input.items()
                },
            }
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

        # Bilinear inputs must be external.  Validate this structural rule once
        # from the declared support instead of asserting inside every traced
        # assembly.
        for i, entry in enumerate(self._units):
            support = entry["support"]
            bilinear_inputs = set(support["E_inputs"]) | set(
                support["F_by_input"]
            )
            for k in bilinear_inputs:
                assert entry["cols"][k][0] == "ext", (
                    f"bilinear input column {k} of unit {i} is internal -- "
                    "cluster is not fusable"
                )
            for input_cols in support["F_by_input"].values():
                for k2 in input_cols:
                    assert entry["cols"][k2][0] == "ext", (
                        "bilinear F term references an internal input -- "
                        "cluster is not fusable"
                    )

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

        # Validate the declared feedthrough graph once.  Runtime matrix values
        # cannot safely decide graph structure: a coefficient may vanish only
        # at the current parameter iterate, and inspecting it would synchronize
        # CUDA during every transformed assembly.
        visiting = set()
        visited = set()
        input_order = []

        def visit_input(i, k):
            node = (i, k)
            if node in visited:
                return
            assert node not in visiting, (
                "algebraic loop in fused feedthrough support -- cluster is not "
                "fusable"
            )
            visiting.add(node)
            kind = self._units[i]["cols"][k]
            if kind[0] == "int":
                j, row = self._sender_rows[(kind[1], kind[2])]
                for kk in self._units[j]["support"]["D_by_row"].get(row, ()):
                    visit_input(j, kk)
            visiting.remove(node)
            visited.add(node)
            input_order.append(node)

        for i, entry in enumerate(self._units):
            for k in range(len(entry["cols"])):
                visit_input(i, k)
        self._input_order = tuple(input_order)
        self.PARAM_NAMES = tuple(
            f"{entry['param_prefix']}.{name}"
            for entry in self._units
            for name in entry["unit"].PARAM_NAMES
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
        dtype = mats[0][0].dtype
        device = mats[0][0].device

        def _expand(t):
            return t if t.shape[0] == n_c else t.expand(n_c, *t.shape[1:])

        # Substitution: per unit, u_i = S_x[i] @ x + S_u[i] @ u_ext.
        # Row cache keyed (unit index, column); cycle-guarded (a cyclic
        # feedthrough chain would be a purely algebraic loop -- not fusable).
        row_cache = {}
        ext_eye = torch.eye(M, dtype=dtype, device=device)
        for i, k in self._input_order:
            kind = units[i]["cols"][k]
            if kind[0] == "ext":
                sx = torch.zeros((n_c, 1, N), dtype=dtype, device=device)
                su = ext_eye[kind[1]].reshape(1, 1, M).expand(n_c, -1, -1)
            else:
                j, row = self._sender_rows[(kind[1], kind[2])]
                _, _, Cj, Dj, _, _ = mats[j]
                oj, nj = units[j]["x_offset"], units[j]["n_states"]
                sx = torch.nn.functional.pad(
                    _expand(Cj)[:, row:row + 1, :],
                    (oj, N - oj - nj),
                )
                su = torch.zeros((n_c, 1, M), dtype=dtype, device=device)
                Dj_row = _expand(Dj)[:, row, :]  # (n_c, m_j)
                for kk in units[j]["support"]["D_by_row"].get(row, ()):
                    col = Dj_row[:, kk]
                    sub_sx, sub_su = row_cache[(j, kk)]
                    sx = sx + col.reshape(n_c, 1, 1) * sub_sx
                    su = su + col.reshape(n_c, 1, 1) * sub_su
            row_cache[(i, k)] = (sx, su)

        S = []  # per unit: (S_x (n_c, m_i, N), S_u (n_c, m_i, M))
        for i, entry in enumerate(units):
            m_i = len(entry["cols"])
            if m_i:
                rows = [row_cache[(i, k)] for k in range(m_i)]
                sx = torch.cat([r[0] for r in rows], dim=1)
                su = torch.cat([r[1] for r in rows], dim=1)
            else:
                sx = torch.zeros((n_c, 0, N), dtype=dtype, device=device)
                su = torch.zeros((n_c, 0, M), dtype=dtype, device=device)
            S.append((sx, su))

        a_parts = []
        b_parts = []
        e_parts = []
        f_parts = []
        for i, entry in enumerate(units):
            Ai, Bi, Ci, Di, Ei, Fi = (_expand(t) for t in mats[i])
            o, n = entry["x_offset"], entry["n_states"]
            sx, su = S[i]
            a_parts.append(
                torch.nn.functional.pad(
                    Ai, (o, N - o - n, o, N - o - n)
                )
                + torch.nn.functional.pad(
                    Bi @ sx, (0, 0, o, N - o - n)
                )
            )
            b_parts.append(
                torch.nn.functional.pad(Bi @ su, (0, 0, o, N - o - n))
            )

            # Bilinear terms must act on external inputs only: substituted
            # (state-valued) bilinear inputs would create quadratic state
            # terms that no LTI block can represent.
            for k, kind in enumerate(entry["cols"]):
                Ek = Ei[:, k]  # (n_c, n, n)
                Fk = Fi[:, k]  # (n_c, n, m_i)
                has_E = k in entry["support"]["E_inputs"]
                f_inputs = entry["support"]["F_by_input"].get(k, ())
                if not has_E and not f_inputs:
                    continue
                j = kind[1]
                if has_E:
                    e_parts.append(
                        ext_eye[j].reshape(1, M, 1, 1)
                        * torch.nn.functional.pad(
                            Ek, (o, N - o - n, o, N - o - n)
                        ).unsqueeze(1)
                    )
                for k2 in f_inputs:
                    kind2 = entry["cols"][k2]
                    col = Fk[:, :, k2]  # (n_c, n)
                    col_joint = torch.nn.functional.pad(
                        col, (o, N - o - n)
                    )
                    f_parts.append(
                        ext_eye[j].reshape(1, M, 1, 1)
                        * col_joint.unsqueeze(1).unsqueeze(-1)
                        * ext_eye[kind2[1]].reshape(1, 1, 1, M)
                    )

        A = torch.stack(a_parts).sum(0)
        B = torch.stack(b_parts).sum(0)
        E = (
            torch.stack(e_parts).sum(0)
            if e_parts
            else torch.zeros((n_c, M, N, N), dtype=dtype, device=device)
        )
        F = (
            torch.stack(f_parts).sum(0)
            if f_parts
            else torch.zeros((n_c, M, N, M), dtype=dtype, device=device)
        )

        # Published outputs: y = (C_i + D_i S_x) x + (D_i S_u) u_ext rows.
        c_rows = []
        d_rows = []
        for i, row in self._out_rows:
            _, _, Ci, Di, _, _ = mats[i]
            o, n = self._units[i]["x_offset"], self._units[i]["n_states"]
            sx, su = S[i]
            Di_row = _expand(Di)[:, row:row + 1, :]  # (n_c, 1, m_i)
            c_rows.append(
                torch.nn.functional.pad(
                    _expand(Ci)[:, row:row + 1, :],
                    (o, N - o - n),
                )
                + Di_row @ sx
            )
            d_rows.append(Di_row @ su)
        C = torch.cat(c_rows, dim=1)
        D = torch.cat(d_rows, dim=1)

        return A, B, C, D, E, F

    # ------------------------------------------------------------------
    # forward / do_step
    # ------------------------------------------------------------------

    def forward(self, x, inputs, params, sample_time, transform_mode=None):
        """Pure one-step of the fused cluster: ``(state, inputs, params) ->
        (new_state, outputs)``.

        ``inputs`` is keyed by the namespaced external port names
        (``"<member_id>.<port>"``); ``params`` by prefixed parameter paths
        (``"<member_key>.<unit>.<name>"``); outputs cover every member output,
        namespaced.  Matrices are cached per params-dict identity (theta-only
        work, done once per theta in a sequential rollout)."""
        if transform_mode:
            matrices = self._assemble(params)
            disc_cache = None
        else:
            cache = getattr(self, "_fwd_mat_cache", None)
            if cache is None or cache[0] is not params or cache[2] != sample_time:
                cache = (params, self._assemble(params), sample_time, {})
                self._fwd_mat_cache = cache
            matrices = cache[1]
            disc_cache = cache[3]
        A, B, C, D, E, F = matrices
        u = torch.stack([inputs[name] for name in self._ext_names], dim=-1)
        x_next, y = bilinear_onestep(
            A,
            B,
            C,
            D,
            E,
            F,
            x,
            u,
            sample_time,
            disc_cache=disc_cache,
            transform_mode=transform_mode,
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
