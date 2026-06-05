# Standard library imports
import os
import unittest

# Local application imports
# Set test flag
import twin4build
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.translator.translator import (
    StepRule,
    AnyPathRule,
    Node,
    NoStepRule,
    OptionalRule,
    SignaturePattern,
    PathRule,
    SetStepRule,
    SetAnyPathRule,
    Translator,
)
from twin4build.utils.uppath import uppath

twin4build._IS_TESTING = True


class TestTranslator(unittest.TestCase):
    def setUp(self):
        """Set up a fresh translator for each test."""
        self.translator = Translator()
        # Third party imports
        from rdflib import URIRef

        # Local application imports
        import twin4build.core as core

        # Set up a very small semantic model
        self.semantic_model = SemanticModel()

        # Define URIs
        base_uri = "http://example.org/test#"
        ahu_uri = URIRef(base_uri + "AHU_1")
        damper1_uri = URIRef(base_uri + "Damper_1")
        damper2_uri = URIRef(base_uri + "Damper_2")
        damper21_uri = URIRef(base_uri + "Damper_21")
        damper22_uri = URIRef(base_uri + "Damper_22")
        room1_uri = URIRef(base_uri + "Room_1")
        room2_uri = URIRef(base_uri + "Room_2")
        sensor1_uri = URIRef(base_uri + "Temperature_Sensor_1")
        sensor2_uri = URIRef(base_uri + "Temperature_Sensor_2")

        # Add types
        self.semantic_model.instance_graph.add(
            (ahu_uri, core.namespace.RDF.type, core.namespace.BRICK.AHU)
        )
        self.semantic_model.instance_graph.add(
            (damper1_uri, core.namespace.RDF.type, core.namespace.BRICK.Damper)
        )
        self.semantic_model.instance_graph.add(
            (damper2_uri, core.namespace.RDF.type, core.namespace.BRICK.Damper)
        )
        self.semantic_model.instance_graph.add(
            (damper21_uri, core.namespace.RDF.type, core.namespace.BRICK.Damper)
        )
        self.semantic_model.instance_graph.add(
            (damper22_uri, core.namespace.RDF.type, core.namespace.BRICK.Damper)
        )
        self.semantic_model.instance_graph.add(
            (room1_uri, core.namespace.RDF.type, core.namespace.BRICK.Room)
        )
        self.semantic_model.instance_graph.add(
            (room2_uri, core.namespace.RDF.type, core.namespace.BRICK.Room)
        )
        self.semantic_model.instance_graph.add(
            (
                sensor1_uri,
                core.namespace.RDF.type,
                core.namespace.BRICK.Temperature_Sensor,
            )
        )
        self.semantic_model.instance_graph.add(
            (
                sensor2_uri,
                core.namespace.RDF.type,
                core.namespace.BRICK.Temperature_Sensor,
            )
        )

        # Add relationships
        self.semantic_model.instance_graph.add(
            (ahu_uri, core.namespace.BRICK.feeds, damper1_uri)
        )
        self.semantic_model.instance_graph.add(
            (ahu_uri, core.namespace.BRICK.feeds, damper2_uri)
        )
        self.semantic_model.instance_graph.add(
            (damper1_uri, core.namespace.BRICK.feeds, room1_uri)
        )
        # Parallel paths to Room 2
        self.semantic_model.instance_graph.add(
            (damper2_uri, core.namespace.BRICK.feeds, damper21_uri)
        )
        self.semantic_model.instance_graph.add(
            (damper2_uri, core.namespace.BRICK.feeds, damper22_uri)
        )
        self.semantic_model.instance_graph.add(
            (damper21_uri, core.namespace.BRICK.feeds, room2_uri)
        )
        self.semantic_model.instance_graph.add(
            (damper22_uri, core.namespace.BRICK.feeds, room2_uri)
        )

        self.semantic_model.instance_graph.add(
            (room1_uri, core.namespace.BRICK.hasPoint, sensor1_uri)
        )
        self.semantic_model.instance_graph.add(
            (room2_uri, core.namespace.BRICK.hasPoint, sensor2_uri)
        )

        self.semantic_model.serialize(filename_instance_graph="test_instance_graph.ttl")

    def test_exact_rule_matching(self):
        """Test matching StepRule rules against the semantic model."""
        # Local application imports
        import twin4build.core as core

        # Define pattern locally
        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)

        sp = SignaturePattern(id="exact_pattern")
        sp.add_rule(
            StepRule(
                subject=node_ahu,
                object=node_damper,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_ahu)

        # Mock system
        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        # Match
        # Use list of classes for systems_
        complete_groups, incomplete = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        # Expect matches (AHU feeds Damper1, AHU feeds Damper2)
        self.assertEqual(len(complete_groups[DummySystem][sp]), 2)

    def test_optional_rule_matching(self):
        """Test matching Optional rules against the semantic model."""
        # Local application imports
        import twin4build.core as core

        # Define pattern with optional node
        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_fan = Node(cls=core.namespace.BRICK.Fan)  # Doesn't exist in our model

        sp = SignaturePattern(id="optional_pattern")
        # AHU exists, Fan doesn't. Optional relation.
        sp.add_rule(
            OptionalRule(
                subject=node_ahu, object=node_fan, predicate=core.namespace.BRICK.feeds
            )
        )
        sp.add_modeled_node(node_ahu)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, incomplete = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        # Should still match AHU, even if Fan is missing
        self.assertEqual(len(complete_groups[DummySystem][sp]), 1)

    def test_single_path_rule_matching(self):
        """Test matching PathRule rules against the semantic model.
        Should find 1 match for AHU -> Room (via Damper1)

        This rule doesnt match paths via Damper_2 -> Damper_21 -> Room2 or Damper_2 -> Damper_22 -> Room2 because the path splits into two paths.
        This matches the indented behavior of PathRule.
        """
        # Local application imports
        import twin4build.core as core

        # AHU -> feeds -> Damper -> feeds -> Room
        # Check AHU -> Room via feeds (hop 2)
        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="single_path_pattern")
        sp.add_rule(
            PathRule(
                subject=node_ahu, object=node_room, predicate=core.namespace.BRICK.feeds
            )
        )
        sp.add_modeled_node(node_ahu)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, incomplete = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )
        self.assertEqual(len(complete_groups[DummySystem][sp]), 1)

    def test_multi_path_rule_matching(self):
        """Test matching AnyPathRule rules against the semantic model.
        Should find 3 matches for AHU -> Room (one via Damper_1, one via Damper_21, one via Damper_22)
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="multi_path_pattern")
        sp.add_rule(
            AnyPathRule(
                subject=node_ahu, object=node_room, predicate=core.namespace.BRICK.feeds
            )
        )
        sp.add_modeled_node(node_ahu)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, incomplete = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        # print the complete groups
        # for group in complete_groups[DummySystem][sp]:
        #     print("\nGroup--------------------------------:")
        #     for sp_subject, sm_subject in group.items():
        #         print(f"{sp_subject.id}: {sm_subject.uri}")

        # Should find 2 matches for Damper_2 (one via Damper_21, one via Damper_22)
        self.assertEqual(len(complete_groups[DummySystem][sp]), 3)

    def test_multi_path_rule_endpoints_only(self):
        """Test AnyPathRule with endpoints_only=True finds unique endpoint matches.
        BFS discovers 2 unique rooms (Room_1 and Room_2) regardless of the number
        of intermediate paths leading to them. The full recursive version finds 3
        because it tracks distinct intermediate paths (Damper_21 vs Damper_22 to Room_2).
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="multi_path_endpoints_only_pattern")
        sp.add_rule(
            AnyPathRule(
                subject=node_ahu,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
                endpoints_only=True,
            )
        )
        sp.add_modeled_node(node_ahu)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, incomplete = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        # 2 unique endpoints: Room_1 and Room_2
        self.assertEqual(len(complete_groups[DummySystem][sp]), 2)

        # Verify each match maps the Room node to a real Room instance
        for mapping in complete_groups[DummySystem][sp]:
            self.assertIsNotNone(mapping[node_room])

    def test_disconnected_merge_skips_multi_hop_rules(self):
        """``_validate_binding_against_merged`` must not single-edge-check
        ``PathRule`` (or ``AnyPathRule``) bindings during a disconnected
        merge.

        Regression for the SAREF building-space pattern where the merge
        rejects a complete match because ``office_supply_damper
        hasFluidSuppliedBy cooling_coil_airside`` only holds via a
        multi-hop chain (damper → port → sensor → coil), which the
        merge validator wrongly checked as a single triple.

        Pattern below mirrors that shape on the existing AHU/Damper/Room
        fixture: ``AHU feeds Damper`` (one hop) and ``AHU feeds Room``
        (multi-hop via Damper).  Phase-1 partials split between the
        scalar Damper and Room candidates, and the disconnected-merge
        path must accept the multi-hop ``AHU --feeds--> Room`` binding
        without demanding a direct triple.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="ahu_damper_room_mixed_pattern")
        sp.add_rule(
            StepRule(
                subject=node_ahu,
                object=node_damper,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_rule(
            PathRule(
                subject=node_ahu,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_ahu)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        # Before the fix this returned 0 because the disconnected merge
        # rejected every (Damper, Room) pairing on the multi-hop edge.
        # The fixture has two reachable rooms (Room_1 via Damper_1 and
        # Room_2 via Damper_2 / Damper_21 / Damper_22) so we expect at
        # least one complete group.
        self.assertGreater(len(complete_groups[DummySystem][sp]), 0)

    def test_disconnected_merge_speculative_preserves_alternative_hypothesis(self):
        """A disconnected merge whose multi-hop rule could not be
        evaluated (peer endpoint unbound everywhere) must NOT consume
        the modeled-node-bearing source group: the original must
        remain available so a later, *evidence-bearing* merge can
        produce the correct complete match.

        Regression for the SAREF building-space sensor pattern on
        ``one_room_example_model.xlsm`` where every non-AHU
        ``Sensor`` instance generated a ``{Sensor=BTA00x,
        Temperature=...}`` Phase-1 seed that the disconnected merge
        absorbed into the SpaceHeater-rooted partial -- locking
        ``node7`` to a sensor that had no ``hasFluidSuppliedBy``
        path back to ``office_supply_damper`` and preventing the
        supply-damper-rooted partial from ever completing the
        pattern.

        The test mirrors that shape on a small BRICK-style fixture:

        * Modeled node = ``Equipment`` (analog of the SAREF
          BuildingSpace).
        * One ``Damper`` directly attached to the Equipment by a
          ``StepRule`` (``isPartOf``) -- analog of the SAREF
          SpaceHeater.
        * One ``Sensor`` reachable from the Equipment by a
          multi-hop ``PathRule`` (``feeds``) via an intermediate
          Damper-like node -- analog of the
          ``Damper -> ... -> Sensor`` chain.
        * Several stray sensors (``Sensor_2..Sensor_5``) whose only
          incident triple is ``rdf:type Sensor`` -- they are not
          reachable from the Equipment via ``feeds`` and would have
          polluted the Equipment partial under the old
          "skip multi-hop validation" disconnected merge.

        A correct matcher rejects any completion that binds
        ``node_sensor`` to a stray sensor (because no path exists)
        and accepts the completion binding ``node_sensor`` to
        ``Sensor_correct`` (because the path exists in the SM).
        """
        # Third party imports
        from rdflib import URIRef

        # Local application imports
        import twin4build.core as core

        sm = SemanticModel()
        base = "http://example.org/sensor_pollution#"
        equipment_uri = URIRef(base + "Equipment_1")
        damper_uri = URIRef(base + "Damper_1")
        port_uri = URIRef(base + "Port_1")
        sensor_correct_uri = URIRef(base + "Sensor_correct")
        stray_uris = [URIRef(base + f"Sensor_stray_{i}") for i in range(4)]

        # Types
        sm.instance_graph.add(
            (equipment_uri, core.namespace.RDF.type, core.namespace.BRICK.AHU)
        )
        sm.instance_graph.add(
            (damper_uri, core.namespace.RDF.type, core.namespace.BRICK.Damper)
        )
        sm.instance_graph.add(
            (port_uri, core.namespace.RDF.type, core.namespace.BRICK.Damper)
        )
        sm.instance_graph.add(
            (
                sensor_correct_uri,
                core.namespace.RDF.type,
                core.namespace.BRICK.Temperature_Sensor,
            )
        )
        for u in stray_uris:
            sm.instance_graph.add(
                (u, core.namespace.RDF.type, core.namespace.BRICK.Temperature_Sensor)
            )

        # Single-hop structural edge: Damper isPartOf Equipment.
        sm.instance_graph.add(
            (damper_uri, core.namespace.BRICK.isPartOf, equipment_uri)
        )

        # Multi-hop chain: Equipment -> Port -> Sensor_correct via ``feeds``.
        # Equipment "feeds" Port "feeds" Sensor_correct, so the PathRule
        # ``Equipment -[feeds]-> Sensor`` is satisfied via Sensor_correct
        # and via Sensor_correct alone (the strays are unreachable).
        sm.instance_graph.add(
            (equipment_uri, core.namespace.BRICK.feeds, port_uri)
        )
        sm.instance_graph.add(
            (port_uri, core.namespace.BRICK.feeds, sensor_correct_uri)
        )

        node_eq = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)
        node_sensor = Node(cls=core.namespace.BRICK.Temperature_Sensor)

        sp = SignaturePattern(id="speculative_pollution_pattern")
        sp.add_rule(
            StepRule(
                subject=node_damper,
                object=node_eq,
                predicate=core.namespace.BRICK.isPartOf,
            )
        )
        sp.add_rule(
            PathRule(
                subject=node_eq,
                object=node_sensor,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_eq)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=sm
        )

        groups = complete_groups[DummySystem][sp]

        # Must produce at least one complete group.  Before the fix,
        # PHASE1 absorbed every stray ``{Sensor=Sensor_stray_i}`` seed
        # into the Equipment partial (a disconnected merge that the
        # pre-fix validator could not catch because the multi-hop rule
        # was simply skipped), locking ``node_sensor`` to a stray and
        # leaving 0 complete groups.
        self.assertGreater(
            len(groups),
            0,
            "speculative disconnected merge must preserve the modeled-node "
            "source group so the correct multi-hop completion can still "
            "form",
        )

        # Every complete group must bind ``node_sensor`` to
        # ``Sensor_correct`` -- not to a stray.  The strays are not
        # reachable from ``Equipment_1`` along any ``feeds`` path, so
        # ``_has_sm_path`` rejects them on positive disproof.
        from twin4build.model.semantic_model.semantic_model import (
            SemanticInstance,
        )

        for mapping in groups:
            sensor_binding = mapping.get(node_sensor)
            self.assertIsNotNone(sensor_binding)
            sensor_uri_str = str(
                sensor_binding.uri
                if isinstance(sensor_binding, SemanticInstance)
                else sensor_binding
            )
            self.assertEqual(
                sensor_uri_str,
                str(sensor_correct_uri),
                f"node_sensor must bind to Sensor_correct, got {sensor_uri_str}",
            )

    def test_initialization(self):
        """Test translator initialization."""
        self.assertIsNotNone(self.translator)
        self.assertIsNotNone(self.translator.sim2sem_map)
        self.assertIsNotNone(self.translator.sem2sim_map)
        self.assertEqual(len(self.translator.sim2sem_map), 0)
        self.assertEqual(len(self.translator.sem2sim_map), 0)

    def test_translate_with_empty_semantic_model(self):
        """Test translation with an empty semantic model."""
        semantic_model = SemanticModel()

        with self.assertRaises(Exception):
            sim_model = self.translator.translate(semantic_model)

    def test_walker_equivalence_legacy_vs_bidirectional(self):
        """The bidirectional walker reaches the same result-set as the
        legacy walker when seeded at an SP node from which the legacy
        forward walk can already discover the full pattern.

        Acts as the regression-oracle equivalence test the PR2 plan
        prescribes: it pins down that the structural change (forward
        + backward iteration with ``visited_sp_edges`` bookkeeping)
        does not silently alter behavior on inputs the legacy walker
        already handles end-to-end.

        Seed: ``node_ahu`` (rule subject) bound to ``AHU_1``.  A pure
        forward walk visits ``node_damper`` via the outgoing edge.
        The bidirectional walker visits the same edge forward and
        marks it visited; the subsequent backward iteration at
        ``node_damper`` finds the same edge in ``visited_sp_edges``
        and skips it -- so the surviving ``result_maps`` set should
        match the legacy walker's exactly.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)

        sp = SignaturePattern(id="walker_equivalence_pattern")
        sp.add_rule(
            StepRule(
                subject=node_ahu,
                object=node_damper,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_ahu)

        seed_sm = next(
            inst
            for inst in self.semantic_model.get_instances_of_type(
                core.namespace.BRICK.AHU
            )
        )

        def canonical(maps):
            """Reduce a list of partial maps to a hashable, order-
            independent canonical form keyed on (sp.id, sm.uri)."""
            out = set()
            for m in maps:
                items = []
                for sp_n, sm_n in m.items():
                    if sm_n is None:
                        continue
                    if isinstance(sm_n, tuple):
                        sm_repr = (
                            "tuple",
                            tuple(sorted(str(getattr(e, "uri", e)) for e in sm_n)),
                        )
                    else:
                        sm_repr = str(getattr(sm_n, "uri", sm_n))
                    items.append((sp_n.id, sm_repr))
                out.add(tuple(sorted(items)))
            return out

        # Run legacy walker
        legacy_maps, _, _, legacy_pruned = Translator._prune_recursive_legacy(
            seed_sm,
            node_ahu,
            [],
            {},
            {},
            sp,
        )
        # Run bidirectional walker (production path)
        bidir_maps, _, _, bidir_pruned = Translator._prune_recursive(
            seed_sm,
            node_ahu,
            [],
            {},
            {},
            sp,
        )

        self.assertEqual(
            legacy_pruned,
            bidir_pruned,
            "legacy and bidirectional walkers disagree on prune outcome "
            f"(legacy_pruned={legacy_pruned}, bidir_pruned={bidir_pruned})",
        )
        self.assertEqual(
            canonical(legacy_maps),
            canonical(bidir_maps),
            "legacy and bidirectional walkers produced different "
            "result_maps on a forward-discoverable pattern; the "
            "bidirectional walker must not silently change behavior on "
            "inputs the legacy walker handles end-to-end",
        )

    def test_backward_walk_steprule(self):
        """Bidirectional walker reaches an SP node via a backward edge.

        Pattern shape::

            node_ahu --feeds--> node_damper   (modeled)

        Seed Phase 1 picks ``node_damper`` (the modeled node) as the
        seed.  Under the legacy outgoing-only walker, walking from
        ``node_damper`` would find no outgoing edges and produce an
        incomplete partial that the merger would have to glue with a
        separately-seeded ``node_ahu`` walk.  Under the bidirectional
        walker, the seed walk follows the inverse adjacency directly
        and produces a complete match in a single pass.

        The SM contains two AHU --feeds--> Damper edges (Damper_1 and
        Damper_2 from :meth:`setUp`), so two complete groups are
        expected.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)

        sp = SignaturePattern(id="backward_steprule_pattern")
        sp.add_rule(
            StepRule(
                subject=node_ahu,
                object=node_damper,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        # Modeled node is the OBJECT endpoint -- forces backward walk
        # from the seed.
        sp.add_modeled_node(node_damper)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            2,
            "expected 2 complete groups (one per AHU-fed damper) reached "
            "by walking backward from node_damper to node_ahu; got "
            f"{len(groups)}",
        )
        for group in groups:
            self.assertIsNotNone(group.get(node_ahu))
            self.assertIsNotNone(group.get(node_damper))

    def test_backward_walk_pathrule_two_hops(self):
        """Bidirectional walker traverses :class:`PathRule` backward.

        Pattern shape::

            node_ahu --feeds*--> node_room   (modeled = node_room)

        With ``node_room`` as the modeled / seed node, the legacy
        outgoing-only walker would have to seed at ``node_ahu``
        (forward) and merge with a separately-seeded ``node_room``
        partial.  Under the parametric :class:`Direction` walker the
        same SP edge is walked backward through inverse adjacency,
        and ``_SinglePath`` honours backward traversal natively.

        With the :meth:`setUp` SM the *forward* single-path test
        finds 1 match (``AHU_1 -> Damper_1 -> Room_1``) and prunes
        the ``Damper_2`` branch because ``Damper_2`` has two forward
        children.  The *backward* walk does not see that branching:
        each step from a ``Room`` to its damper to ``AHU_1`` has
        cardinality 1 in inverse adjacency.  Three backward paths
        therefore complete:

        - ``Room_1 -> Damper_1 -> AHU_1``
        - ``Room_2 -> Damper_21 -> Damper_2 -> AHU_1``
        - ``Room_2 -> Damper_22 -> Damper_2 -> AHU_1``

        The asymmetry vs. forward (1 vs. 3) confirms the walker is
        actually iterating the inverse direction and not silently
        falling back to forward.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="backward_pathrule_two_hops_pattern")
        sp.add_rule(
            PathRule(
                subject=node_ahu,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_room)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            3,
            "expected 3 backward PathRule matches (Room_1 + 2x Room_2 via "
            f"Damper_21 / Damper_22); got {len(groups)}",
        )
        for group in groups:
            self.assertIsNotNone(group.get(node_ahu))
            self.assertIsNotNone(group.get(node_room))

    def test_backward_walk_anypathrule_branching(self):
        """Bidirectional walker traverses :class:`AnyPathRule` backward.

        Pattern shape::

            node_ahu --feeds*--> node_room   (modeled = node_room, AnyPathRule)

        :class:`AnyPathRule` (multi-path) tolerates branching at every
        hop, in either direction.  Forward-seeded the existing
        :meth:`test_multi_path_rule_matching` produces 3 matches
        (Damper_1 / Damper_21 / Damper_22 endpoints from AHU_1).

        Backward-seeded at ``node_room`` the walker enumerates every
        room that has an inverse path back to an AHU.  Both rooms in
        :meth:`setUp` reach AHU_1 by inverse adjacency:

        - ``Room_1`` via ``Damper_1``
        - ``Room_2`` via ``Damper_21 -> Damper_2`` and via
          ``Damper_22 -> Damper_2`` (AnyPathRule preserves both
          intermediates because branching is allowed).

        Three backward matches confirm the walker is actually
        following ``predicate_subject_pairs`` and the multi-path
        cardinality gate fires symmetrically in inverse adjacency.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="backward_anypathrule_pattern")
        sp.add_rule(
            AnyPathRule(
                subject=node_ahu,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_room)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            3,
            "expected 3 backward AnyPathRule matches (Room_1 + 2x Room_2 "
            f"via Damper_21 / Damper_22); got {len(groups)}",
        )
        for group in groups:
            self.assertIsNotNone(group.get(node_ahu))
            self.assertIsNotNone(group.get(node_room))

    def test_pathrule_two_hop_with_downstream_step_rule(self):
        """Regression: ``PathRule``'s multi-hop ``_SinglePath`` branch must
        be able to fire its *downstream* :class:`StepRule` after the
        sibling :class:`StepRule` branch (1-hop) failed at a
        *different* SM subject.

        Pattern shape::

            node_damper --feeds*--> node_sensor --observes--> node_temp
                          (PathRule)              (StepRule)

        SM::

            damper -[feeds]-> sensor_pressure -[feeds]-> sensor_temp
            sensor_temp -[observes]-> temperature
            (sensor_pressure has NO observes-edge to a Temperature)

        Two ``PathRule`` branches fire from ``damper``:

        - The 1-hop ``StepRule`` branch matches ``sensor_pressure``.
          The walker recurses at ``sensor_pressure`` and tries the
          downstream ``observes Temperature`` ``StepRule`` -- which
          fails because ``sensor_pressure`` has no such edge.  This
          attempt marks the SP edge ``(node_sensor, observes,
          node_temp)`` visited.
        - The 2-hop ``_SinglePath`` branch then reaches
          ``sensor_temp``.  The walker recurses at ``sensor_temp``
          and must be able to re-evaluate the downstream
          ``observes Temperature`` edge -- *from a different SM
          subject* -- and bind ``node_temp`` to ``temperature``.

        Before the fix, ``visited_sp_edges`` was keyed by the SP edge
        only, so the failed first attempt poisoned the second:
        ``(node_sensor, observes, node_temp)`` was already in the set
        and the walker could not re-fire the downstream rule from
        ``sensor_temp``.  ``node_temp`` therefore stayed unbound and
        the matcher produced **0** complete groups.

        Re-keying ``visited_sp_edges`` by ``(sm_subject, edge_key)``
        scopes the cycle-prevention to the (SM-subject, SP-edge)
        pair, so the same SP edge can be re-evaluated when entered
        from a different SM subject.  The matcher then produces
        **1** complete group bound to ``Sensor=sensor_temp,
        Temperature=temperature``.
        """
        # Third party imports
        from rdflib import URIRef

        # Local application imports
        import twin4build.core as core
        from twin4build.model.semantic_model.semantic_model import (
            SemanticInstance,
        )

        sm = SemanticModel()
        base = "http://example.org/two_hop_pathrule#"
        damper_uri = URIRef(base + "Damper_1")
        sensor_pressure_uri = URIRef(base + "Sensor_pressure")
        sensor_temp_uri = URIRef(base + "Sensor_temp")
        temp_uri = URIRef(base + "Temperature_1")

        sm.instance_graph.add(
            (damper_uri, core.namespace.RDF.type, core.namespace.BRICK.Damper)
        )
        sm.instance_graph.add(
            (
                sensor_pressure_uri,
                core.namespace.RDF.type,
                core.namespace.BRICK.Sensor,
            )
        )
        sm.instance_graph.add(
            (sensor_temp_uri, core.namespace.RDF.type, core.namespace.BRICK.Sensor)
        )
        sm.instance_graph.add(
            (temp_uri, core.namespace.RDF.type, core.namespace.BRICK.Temperature)
        )

        # Two-hop chain: damper -[feeds]-> sensor_pressure -[feeds]-> sensor_temp
        sm.instance_graph.add(
            (damper_uri, core.namespace.BRICK.feeds, sensor_pressure_uri)
        )
        sm.instance_graph.add(
            (sensor_pressure_uri, core.namespace.BRICK.feeds, sensor_temp_uri)
        )
        # Only sensor_temp observes a Temperature.  sensor_pressure has
        # no outgoing observes triple at all, so the StepRule branch
        # of the PathRule fails at sensor_pressure.
        sm.instance_graph.add(
            (sensor_temp_uri, core.namespace.BRICK.observes, temp_uri)
        )

        node_damper = Node(cls=core.namespace.BRICK.Damper)
        node_sensor = Node(cls=core.namespace.BRICK.Sensor)
        node_temp = Node(cls=core.namespace.BRICK.Temperature)

        sp = SignaturePattern(id="pathrule_two_hop_downstream_step_pattern")
        sp.add_rule(
            PathRule(
                subject=node_damper,
                object=node_sensor,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_rule(
            StepRule(
                subject=node_sensor,
                object=node_temp,
                predicate=core.namespace.BRICK.observes,
            )
        )
        sp.add_modeled_node(node_damper)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=sm
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            1,
            "expected exactly 1 complete match (Sensor=sensor_temp, "
            "Temperature=temperature) -- the PathRule's _SinglePath "
            "branch must reach sensor_temp and complete the downstream "
            "StepRule even after the StepRule branch failed at "
            f"sensor_pressure; got {len(groups)}",
        )

        for mapping in groups:
            sensor_binding = mapping.get(node_sensor)
            temp_binding = mapping.get(node_temp)
            self.assertIsNotNone(
                sensor_binding,
                "node_sensor must be bound when the downstream StepRule "
                "completes via _SinglePath",
            )
            self.assertIsNotNone(
                temp_binding,
                "node_temp must be bound -- the downstream StepRule must "
                "fire from sensor_temp even though it was rejected from "
                "sensor_pressure earlier in the walk",
            )
            sensor_uri_str = str(
                sensor_binding.uri
                if isinstance(sensor_binding, SemanticInstance)
                else sensor_binding
            )
            temp_uri_str = str(
                temp_binding.uri
                if isinstance(temp_binding, SemanticInstance)
                else temp_binding
            )
            self.assertEqual(
                sensor_uri_str,
                str(sensor_temp_uri),
                "node_sensor must bind to sensor_temp (the only sensor "
                "with a Temperature observation), not sensor_pressure",
            )
            self.assertEqual(
                temp_uri_str,
                str(temp_uri),
                "node_temp must bind to the SM Temperature instance "
                "reached via sensor_temp",
            )

    def test_visited_edges_still_block_same_sm_cycle(self):
        """Re-keying ``visited_sp_edges`` by ``(sm_subject, edge_key)``
        must still prevent the matcher from re-traversing the same SP
        edge from the same SM subject.

        Pattern: a self-loop ``node_a --feeds--> node_a``.  SM: a
        single :class:`BRICK.AHU` instance with a self-loop on
        ``BRICK.feeds``.  The walker enters at the seed, follows the
        forward edge back to the same SM instance, and the recursive
        entry must short-circuit on ``(a, (node_a, feeds, node_a))``
        already being in ``visited_sp_edges``.

        Asserts the matcher terminates (no infinite recursion) and
        produces exactly one complete group binding ``node_a`` to the
        single SM AHU -- demonstrating that the per-(sm_subject,
        edge) keying still vetoes same-subject same-edge re-entry.
        """
        # Third party imports
        from rdflib import URIRef

        # Local application imports
        import twin4build.core as core

        sm = SemanticModel()
        base = "http://example.org/self_loop#"
        a_uri = URIRef(base + "AHU_self_loop")

        sm.instance_graph.add(
            (a_uri, core.namespace.RDF.type, core.namespace.BRICK.AHU)
        )
        sm.instance_graph.add((a_uri, core.namespace.BRICK.feeds, a_uri))

        node_a = Node(cls=core.namespace.BRICK.AHU)

        sp = SignaturePattern(id="self_loop_cycle_pattern")
        sp.add_rule(
            StepRule(
                subject=node_a,
                object=node_a,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_a)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        # Must not infinite-loop; must produce exactly one match.
        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=sm
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            1,
            "expected exactly 1 complete group for the self-loop SP "
            "(node_a --feeds--> node_a) -- the (sm_subject, edge_key) "
            "keying must still block re-traversal of the same SP edge "
            f"from the same SM subject; got {len(groups)}",
        )
        for mapping in groups:
            self.assertIsNotNone(
                mapping.get(node_a),
                "node_a must be bound to the single SM AHU instance",
            )

    def test_setsteprule_forward_through_bidirectional_broadcast(self):
        """Forward-seeded :class:`SetStepRule` produces a parallel-tuple
        binding via the bidirectional broadcast helper.

        This exercises both the parametric :meth:`SetStepRule.apply`
        (forward direction) and the new bidirectional
        :meth:`Translator.__broadcast_recurse` helper.  The broadcast
        threads ``visited_sp_edges`` per element so the walker does
        not oscillate back through the SetStepRule edge while
        broadcasting per-element recursions.

        SM (extends the setUp): ``Room_1 -> hasPoint -> Sensor1`` and
        ``Room_2 -> hasPoint -> Sensor2``.  Pattern binds
        ``node_room`` (modeled, scalar) -> ``hasPoint`` ->
        ``node_sensors`` (set-bound).  Two complete groups are
        expected, one per room, each with a single-element tuple
        binding for ``node_sensors``.
        """
        # Local application imports
        import twin4build.core as core

        node_room = Node(cls=core.namespace.BRICK.Room)
        node_sensors = Node(cls=core.namespace.BRICK.Temperature_Sensor)

        sp = SignaturePattern(id="setsteprule_forward_pattern")
        sp.add_rule(
            SetStepRule(
                subject=node_room,
                object=node_sensors,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )
        sp.add_modeled_node(node_room)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            2,
            "expected one complete group per BRICK.Room (each with a "
            f"single-element sensor tuple); got {len(groups)}",
        )
        for group in groups:
            self.assertIsNotNone(group.get(node_room))
            sensors_binding = group.get(node_sensors)
            self.assertIsInstance(
                sensors_binding,
                tuple,
                "node_sensors binding produced by SetStepRule must be a tuple",
            )
            self.assertEqual(
                len(sensors_binding),
                1,
                f"each room has exactly one sensor; got tuple of length "
                f"{len(sensors_binding)}",
            )

    def test_setanypathrule_forward_multi_hop_aggregation(self):
        """Forward-seeded :class:`SetAnyPathRule` aggregates every
        multi-hop endpoint under the predicate into a single tuple
        binding.

        SM (from :meth:`setUp`): the AHU reaches ``Room_1`` (one hop
        through ``Damper_1``) and ``Room_2`` (two hops through
        ``Damper_2 -> Damper_21|22``) under ``BRICK.feeds``.  The
        ``SetAnyPathRule`` BFS uses :meth:`Direction.sm_adj` -- in
        forward direction this is ``predicate_object_pairs`` -- and
        collects every reachable :class:`BRICK.Room` endpoint into one
        canonical tuple of length 2.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_rooms = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="setanypath_forward_pattern")
        sp.add_rule(
            SetAnyPathRule(
                subject=node_ahu,
                object=node_rooms,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_ahu)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            1,
            "expected one complete group (one AHU, one tuple of every "
            f"reachable room); got {len(groups)}",
        )
        rooms_binding = groups[0].get(node_rooms)
        self.assertIsInstance(
            rooms_binding,
            tuple,
            "SetAnyPathRule must produce a tuple binding on its object",
        )
        self.assertEqual(
            len(rooms_binding),
            2,
            "expected both Room_1 and Room_2 in the aggregated tuple; "
            f"got tuple of length {len(rooms_binding)}",
        )
        room_uris = {str(r.uri).rsplit("#", 1)[-1] for r in rooms_binding}
        self.assertSetEqual(
            room_uris,
            {"Room_1", "Room_2"},
            "tuple should contain both Room_1 and Room_2 (deduplicated "
            f"despite two parallel paths to Room_2); got {room_uris}",
        )

    def test_setanypathrule_apply_backward_direct(self):
        """:meth:`SetAnyPathRule.apply` body is direction-parametric
        with asymmetric forward/backward emission.

        Forward emits one tuple-binding pair (set-binding on the SP
        object); backward emits *scalar* per-endpoint pairs -- one pair
        per reachable near reached through inverse adjacency.  The
        scalar shape is required because the SP subject of a
        ``SetAnyPathRule`` is scalar by construction; bundling
        backward-reached nears into a singleton tuple at a scalar SP
        node would force the walker's tuple-handling branch onto a
        scalar position and confuse the broadcast aggregator.  The
        full set binding on the SP object is restored by the natural
        FORWARD firing at the near reached via this backward walk
        (see :meth:`SetAnyPathRule.apply` docstring).

        This test seeds at a :class:`BRICK.Room` instance and asserts
        the BFS reaches the AHU through inverse-feeds edges,
        producing one scalar pair per reachable AHU.
        """
        # Local application imports
        from twin4build.translator.translator import BACKWARD, Predicate, StepRule
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_rooms = Node(cls=core.namespace.BRICK.Room)
        sp = SignaturePattern(id="setanypath_backward_unit_pattern")
        sp.add_rule(
            SetAnyPathRule(
                subject=node_ahu,
                object=node_rooms,
                predicate=Predicate(core.namespace.BRICK.feeds),
            )
        )

        # The Predicate equality-hash semantics make ruleset key lookup
        # by reconstructed predicate brittle; resolve the rule instance
        # by scanning the ruleset values for the SetAnyPathRule.
        candidates = [
            r for r in sp.ruleset.values() if isinstance(r, SetAnyPathRule)
        ]
        self.assertEqual(
            len(candidates),
            1,
            f"expected one SetAnyPathRule registered in ruleset; got {len(candidates)}",
        )
        rule = candidates[0]

        room_sm = next(
            iter(
                self.semantic_model.get_instances_of_type(core.namespace.BRICK.Room)
            )
        )

        pairs, rule_applies, _, _ = rule.apply(
            room_sm,
            [room_sm],
            sp.ruleset,
            direction=BACKWARD,
        )
        self.assertTrue(
            rule_applies,
            "backward BFS from a Room must reach the AHU through "
            "inverse-feeds adjacency",
        )
        self.assertEqual(
            len(pairs),
            1,
            "SetAnyPathRule emits one scalar pair per reachable near "
            "in backward direction; expected exactly one (single AHU "
            f"reachable via inverse-feeds); got {len(pairs)}",
        )
        _, sm_near, sp_target, kind, _ = pairs[0]
        self.assertIs(
            kind,
            StepRule,
            "backward set-rules emit scalar pairs tagged StepRule so "
            "the walker takes the scalar (non-broadcast) recursion "
            "branch; tuple binding is restored later by the forward "
            "firing at the near",
        )
        self.assertIs(
            sp_target,
            node_ahu,
            "backward-direction far node must be the SP subject "
            "(node_ahu); got a different SP node",
        )
        self.assertNotIsInstance(
            sm_near,
            tuple,
            "backward set-rule emission must be scalar (the matched "
            "SM near), not a tuple binding",
        )
        ahu_uri_short = str(sm_near.uri).rsplit("#", 1)[-1]
        self.assertEqual(
            ahu_uri_short,
            "AHU_1",
            f"backward BFS must terminate at AHU_1; got {ahu_uri_short}",
        )

    def test_setsteprule_backward_through_bidirectional_walker(self):
        """Backward-seeded :class:`SetStepRule` recovers the full tuple
        binding via the natural FORWARD firing at the near.

        Pattern: ``node_room (modeled, scalar) <-- StepRule(hasPoint) --
        node_sensor (scalar)`` plus
        ``node_ahu (scalar) -- SetStepRule(feeds) --> node_dampers
        (set-bound)``, with ``node_dampers <-- StepRule(feeds) --
        node_room`` linking the two halves.  Seeding at the modeled
        Room walks backward through ``StepRule(node_dampers, node_room,
        feeds)`` to a damper, then backward through the
        ``SetStepRule(node_ahu, node_dampers, feeds)`` to ``AHU_1``.
        The matcher's recursion at ``(AHU_1, node_ahu)`` then fires
        the ``SetStepRule`` *forward* and the broadcast helper
        re-establishes the full ``node_dampers`` tuple binding on the
        SP object -- which, after broadcast filter semantics, retains
        only the dampers whose downstream ``StepRule`` to a room is
        satisfied (see ``__broadcast_recurse`` filter semantics).

        Verifies:
          1. The seed walk *reaches* ``node_ahu`` despite no forward
             path from the seed Room (the only route is backward
             through the set-rule).  Pre-fix, the readiness gate
             skipped backward dispatch on ``SetStepRule``, leaving
             ``node_ahu`` unbound and the match incomplete.
          2. ``node_dampers`` ends up as a tuple (set-bound binding)
             rather than a stray scalar from the backward walk's
             intermediate scalar binding.
          3. ``node_ahu`` binds to a single ``AHU`` instance (scalar).
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_dampers = Node(cls=core.namespace.BRICK.Damper)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="setsteprule_backward_pattern")
        sp.add_rule(
            SetStepRule(
                subject=node_ahu,
                object=node_dampers,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_rule(
            StepRule(
                subject=node_dampers,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_room)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )
        groups = complete_groups[DummySystem][sp]

        # In the test SM, only Damper_1 directly feeds Room_1 (Damper_2
        # feeds Damper_21/22 which then feed Room_2 -- that's *two*
        # hops, not directly satisfied by ``StepRule(feeds)``).  The
        # broadcast filter prunes Damper_2 from the surviving tuple.
        # We therefore expect exactly one complete group: AHU_1 ->
        # (Damper_1,) -> Room_1.
        self.assertEqual(
            len(groups),
            1,
            "expected one complete group (AHU_1 -> (Damper_1,) -> "
            f"Room_1); got {len(groups)}",
        )
        group = groups[0]

        # The set-rule subject must be bound (the WHOLE point of
        # backward set-rule support).  Pre-fix, this would be None.
        self.assertIsNotNone(
            group.get(node_ahu),
            "node_ahu must be bound after backward set-rule traversal "
            "(the modeled Room can only reach the AHU by walking "
            "backward through the SetStepRule edge)",
        )
        self.assertNotIsInstance(
            group.get(node_ahu),
            tuple,
            "node_ahu is the scalar SP subject of the SetStepRule -- "
            "it must remain scalar (not a singleton tuple) after the "
            "backward traversal",
        )

        # Set-bound node must hold a tuple binding (re-established by
        # the forward firing at the AHU after the backward walk).
        dampers_binding = group.get(node_dampers)
        self.assertIsInstance(
            dampers_binding,
            tuple,
            "node_dampers (set-bound) must hold a tuple binding "
            "after the forward firing at AHU re-establishes the set",
        )
        # After broadcast filter, only Damper_1 survives (Damper_2
        # has no direct feeds -> Room edge).
        damper_uris = sorted(str(d.uri).rsplit("#", 1)[-1] for d in dampers_binding)
        self.assertEqual(
            damper_uris,
            ["Damper_1"],
            "after broadcast filter semantics, only the dampers whose "
            "downstream StepRule(feeds, room) is satisfied survive; "
            f"got {damper_uris}",
        )

        ahu_uri_short = str(group.get(node_ahu).uri).rsplit("#", 1)[-1]
        self.assertEqual(
            ahu_uri_short,
            "AHU_1",
            f"node_ahu must bind to AHU_1; got {ahu_uri_short}",
        )

    def test_setanypathrule_backward_through_bidirectional_walker(self):
        """Backward-seeded :class:`SetAnyPathRule` recovers the full
        multi-hop tuple binding via the natural FORWARD firing at the
        near.

        Pattern: ``node_ahu (scalar) -- SetAnyPathRule(feeds) -->
        node_rooms (set-bound)`` with ``node_room_anchor (modeled,
        scalar)`` linked via auto-broadcast.  Seeding at the modeled
        Room walks backward through the multi-hop inverse-feeds chain
        (``Room_2 <- Damper_22 <- Damper_2 <- AHU_1`` *or*
        ``Room_2 <- Damper_21 <- Damper_2 <- AHU_1`` *or*
        ``Room_1 <- Damper_1 <- AHU_1``) to ``AHU_1``, then the
        forward firing of the ``SetAnyPathRule`` at ``AHU_1``
        rebuilds the full multi-hop endpoint tuple
        ``(Room_1, Room_2)``.

        Verifies the multi-hop counterpart of
        :meth:`test_setsteprule_backward_through_bidirectional_walker`:
        the BFS-based ``SetAnyPathRule.apply`` body emits one scalar
        pair per *reachable* near in backward direction (here a
        single ``AHU_1``), and the matcher's natural forward
        re-firing reconstructs the full set binding through the
        BFS frontier.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_rooms = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="setanypath_backward_pattern")
        sp.add_rule(
            SetAnyPathRule(
                subject=node_ahu,
                object=node_rooms,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        # Modeled at the set-bound side: this forces the seed to be
        # an individual Room, exercising the backward set-rule
        # traversal contract.  Without backward dispatch on
        # SetAnyPathRule, the seed walk would produce only an
        # incomplete partial mapping (node_ahu = None).
        sp.add_modeled_node(node_rooms)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )
        groups = complete_groups[DummySystem][sp]

        # Exactly one canonical complete group is expected: AHU_1
        # paired with the full set of reachable rooms.  Multiple
        # seeds (Room_1, Room_2) collapse to one mapping after Phase 6
        # dedupe; multiple parallel paths to Room_2 are deduplicated
        # by SetAnyPathRule's BFS endpoint set.
        self.assertEqual(
            len(groups),
            1,
            "expected one canonical complete group (AHU_1 -> all "
            f"reachable rooms); got {len(groups)}",
        )
        group = groups[0]

        # The set-rule subject must be bound -- this is the regression
        # marker for backward dispatch on SetAnyPathRule.
        self.assertIsNotNone(
            group.get(node_ahu),
            "node_ahu must be bound after backward SetAnyPathRule "
            "traversal (only reachable from a Room seed by walking "
            "backward through the multi-hop inverse-feeds chain)",
        )
        self.assertNotIsInstance(
            group.get(node_ahu),
            tuple,
            "scalar SP subject of a SetAnyPathRule must stay scalar",
        )

        rooms_binding = group.get(node_rooms)
        self.assertIsInstance(
            rooms_binding,
            tuple,
            "node_rooms (set-bound) must hold a tuple binding "
            "re-established by the forward firing at AHU",
        )
        room_uris = {str(r.uri).rsplit("#", 1)[-1] for r in rooms_binding}
        self.assertSetEqual(
            room_uris,
            {"Room_1", "Room_2"},
            "the forward re-firing at AHU_1 must rebuild the full set "
            f"of reachable rooms; got {room_uris}",
        )

    def test_nosteprule_apply_symmetric_veto(self):
        """:meth:`NoStepRule.apply` enforces the absence-of-edge veto
        symmetrically across forward and backward seed directions.

        The veto applies to the triple ``Room -hasPoint-> Sensor``.
        When :meth:`NoStepRule.apply` is called with a Room as
        ``sm_subject`` and the corresponding Sensor as the SM
        candidate (forward), or with the Sensor as ``sm_subject`` and
        the Room as the SM candidate (backward), both arms must
        report ``rule_applies == False`` -- i.e. the veto fires
        regardless of seed direction.

        The forward case is the legacy semantic; the backward case
        relies on the parametric ``direction.far(self)`` lookup
        introduced in B'-PR2.6 (without it, the rule would be
        comparing the SP subject's class against a backward-seeded SM
        object and silently let a forbidden triple through).
        """
        # Local application imports
        from twin4build.translator.translator import BACKWARD, FORWARD
        import twin4build.core as core

        node_room = Node(cls=core.namespace.BRICK.Room)
        node_sensor = Node(cls=core.namespace.BRICK.Temperature_Sensor)

        sp = SignaturePattern(id="nosteprule_symmetric_pattern")
        sp.add_rule(
            NoStepRule(
                subject=node_room,
                object=node_sensor,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )

        rule = next(
            r for r in sp.ruleset.values() if isinstance(r, NoStepRule)
        )

        room_sm = next(
            iter(
                self.semantic_model.get_instances_of_type(core.namespace.BRICK.Room)
            )
        )
        sensor_sm = next(
            iter(
                self.semantic_model.get_instances_of_type(
                    core.namespace.BRICK.Temperature_Sensor
                )
            )
        )

        # Forward: Room as SM near, Sensor as SM far candidate.
        # Sensor IS a Temperature_Sensor (== far_node.cls), so the
        # forbidden triple exists -- veto fires (rule_applies=False).
        _, fwd_applies, _, _ = rule.apply(
            room_sm,
            [sensor_sm],
            sp.ruleset,
            candidate_maps=[],
            direction=FORWARD,
        )
        self.assertFalse(
            fwd_applies,
            "forward NoStepRule must veto when an SM neighbor "
            "satisfies the forbidden far class",
        )

        # Backward: Sensor as SM near, Room as SM far candidate.
        # Room IS a Room (== direction.far(self).cls in BACKWARD), so
        # the forbidden triple still exists from the inverse view --
        # the symmetric veto must fire.
        _, bwd_applies, _, _ = rule.apply(
            sensor_sm,
            [room_sm],
            sp.ruleset,
            candidate_maps=[],
            direction=BACKWARD,
        )
        self.assertFalse(
            bwd_applies,
            "backward NoStepRule must veto symmetrically: an SM "
            "neighbor reached via inverse-predicate that satisfies "
            "the SP subject class is the same forbidden triple as "
            "the forward case",
        )

    def test_diagnostic_dump_surfaces_direction_labels(self):
        """The TWIN4BUILD_MATCH_DIAG_FILE diagnostic dump labels every
        walker decision with an explicit ``dir=forward`` or
        ``dir=backward`` token.

        Without direction labels, debugging a bidirectional matcher
        run becomes ambiguous: a "PRUNE" event could be a forward or
        backward sweep failing, and the pruning reason
        (missing-predicate, no-match, gate-skip) would then have to
        be cross-referenced against pattern topology to recover the
        direction.  The PR5 diagnostic refresh emits ``[WALKER]``
        entries with ``dir=...`` on every dispatch and prune so the
        log is self-describing.

        The test sets ``TWIN4BUILD_MATCH_DIAG_FILE`` for the duration
        of one matching call, runs the AHU/Damper backward-walk
        pattern (which fires both directions for every seed), and
        asserts the dump contains lines tagged with both
        ``dir=forward`` and ``dir=backward`` and at least one
        ``[WALKER]`` event.
        """
        # Standard library imports
        import importlib
        import os
        import tempfile

        # Local application imports
        import twin4build.core as core
        from twin4build.translator import translator as translator_mod

        # AHU --feeds-> Damper --feeds-> Room.  Modeled = node_damper
        # so the seed lands at SM Dampers.  At each Damper seed the
        # walker enumerates both outgoing edges (forward to Room) and
        # incoming edges (backward to AHU); both directions therefore
        # produce ``[WALKER]`` diagnostic entries with the
        # corresponding ``dir=`` label.
        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="diag_direction_label_pattern")
        sp.add_rule(
            StepRule(
                subject=node_ahu,
                object=node_damper,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_rule(
            StepRule(
                subject=node_damper,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_damper)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "matcher_diag.log")
            os.environ["TWIN4BUILD_MATCH_DIAG_FILE"] = log_path
            os.environ["TWIN4BUILD_MATCH_DIAG_PATTERN"] = (
                "diag_direction_label_pattern"
            )
            try:
                # Reload the module-level diag-path/filter cache (the
                # globals are read from the environment at module
                # import time, so the test must explicitly refresh
                # them after setting the envs).
                translator_mod._MATCH_DIAG_PATH = log_path
                translator_mod._MATCH_DIAG_PATTERN_FILTER = (
                    "diag_direction_label_pattern"
                )
                translator_mod._MATCH_DIAG_FH = None

                Translator._match_patterns(
                    systems_=[DummySystem], semantic_model=self.semantic_model
                )

                # Close the file handle so all writes are flushed.
                fh = translator_mod._MATCH_DIAG_FH
                if fh is not None:
                    fh.close()
                    translator_mod._MATCH_DIAG_FH = None

                with open(log_path, "r", encoding="utf-8") as f:
                    log_text = f.read()
            finally:
                # Reset env + module-level state so other tests are
                # not impacted by the diag path.
                os.environ.pop("TWIN4BUILD_MATCH_DIAG_FILE", None)
                os.environ.pop("TWIN4BUILD_MATCH_DIAG_PATTERN", None)
                translator_mod._MATCH_DIAG_PATH = None
                translator_mod._MATCH_DIAG_PATTERN_FILTER = None
                if translator_mod._MATCH_DIAG_FH is not None:
                    try:
                        translator_mod._MATCH_DIAG_FH.close()
                    except Exception:
                        pass
                    translator_mod._MATCH_DIAG_FH = None

        self.assertIn(
            "[WALKER]",
            log_text,
            "diagnostic dump must surface walker-scope events when "
            "TWIN4BUILD_MATCH_DIAG_FILE is enabled",
        )
        self.assertIn(
            "dir=forward",
            log_text,
            "walker events must be tagged with dir=forward for the "
            "forward sweep so the bidirectional walker is "
            "self-describing in the dump",
        )
        self.assertIn(
            "dir=backward",
            log_text,
            "walker events must be tagged with dir=backward for the "
            "backward sweep so the bidirectional walker is "
            "self-describing in the dump",
        )

    def test_merger_noop_on_connected_pattern(self):
        """A connected SP graph (single WCC) needs no merger after the
        bidirectional walker.

        After the parametric direction refactor (PR2.1-PR2.6) the
        unified bidirectional walker fills every weakly-connected
        component (WCC) of the SP graph from its single Phase-1 seed.
        :meth:`Translator._merge_incomplete_groups` therefore acts as
        a no-op on connected patterns: any partials it receives are
        returned unchanged, no new ``complete_matches`` are produced
        by the merger.

        The test seeds the matcher with a deliberately under-explored
        partial mapping for a connected AHU/Damper/Room pattern and
        asserts both invariants:

        1. ``incomplete_matches`` is returned unchanged (length and
           identity-preserving).
        2. ``complete_matches`` is not extended by the merger.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)
        node_room = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="merger_noop_connected_pattern")
        sp.add_rule(
            StepRule(
                subject=node_ahu,
                object=node_damper,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_rule(
            PathRule(
                subject=node_ahu,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_ahu)

        wccs = sp.weakly_connected_components()
        self.assertEqual(
            len(wccs),
            1,
            f"prerequisite: SP graph must be connected (1 WCC); got {len(wccs)} "
            "WCCs -- if this fails, the test fixture has been altered",
        )

        # Synthetic incomplete partials: each one binds only a
        # subset of nodes.  These would normally be the merger's
        # input.  After the bidirectional walker the runtime
        # produces complete matches directly, so the merger receives
        # at most these synthetic partials and -- per the PR4
        # invariant -- returns them unchanged.
        ahu_sm = next(
            iter(
                self.semantic_model.get_instances_of_type(core.namespace.BRICK.AHU)
            )
        )
        damper_sm = next(
            iter(
                self.semantic_model.get_instances_of_type(core.namespace.BRICK.Damper)
            )
        )
        partial_ahu = {n: None for n in sp.nodes}
        partial_ahu[node_ahu] = ahu_sm
        partial_damper = {n: None for n in sp.nodes}
        partial_damper[node_damper] = damper_sm

        incomplete_matches = [partial_ahu, partial_damper]
        complete_matches: list = []
        complete_count_before = len(complete_matches)
        incomplete_count_before = len(incomplete_matches)

        result = Translator._merge_incomplete_groups(
            incomplete_matches, complete_matches, sp
        )

        self.assertEqual(
            len(result),
            incomplete_count_before,
            "connected-pattern merger must return incomplete_matches unchanged: "
            f"got {len(result)} entries, expected {incomplete_count_before}",
        )
        self.assertIs(
            result,
            incomplete_matches,
            "connected-pattern merger must preserve list identity (no-op fast path)",
        )
        self.assertEqual(
            len(complete_matches),
            complete_count_before,
            "connected-pattern merger must NOT extend complete_matches: "
            f"got {len(complete_matches)} entries, expected "
            f"{complete_count_before}",
        )

    def test_nosteprule_walker_backward_seed_veto(self):
        """The bidirectional walker dispatches :class:`NoStepRule`
        backward through the readiness gate.

        Pattern shape::

            node_room --hasPoint--> node_sensor   (NoStepRule)

        Modeled = ``node_sensor`` forces the seed at SM Sensors.
        Every Sensor in the SM is reachable backward through
        ``hasPoint`` to its parent Room, which satisfies the
        forbidden ``BRICK.Room`` class.  The backward dispatch of
        :meth:`NoStepRule.apply` reports rule_applies=False, and the
        walker prunes -- so zero complete groups are produced.

        Without the readiness gate change in B'-PR2.6 (admitting
        ``NoStepRule``), the backward edge would be silently skipped
        and the walker would happily seed every Sensor without ever
        evaluating the veto.
        """
        # Local application imports
        import twin4build.core as core

        node_room = Node(cls=core.namespace.BRICK.Room)
        node_sensor = Node(cls=core.namespace.BRICK.Temperature_Sensor)

        sp = SignaturePattern(id="nosteprule_backward_walker_pattern")
        sp.add_rule(
            NoStepRule(
                subject=node_room,
                object=node_sensor,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )
        sp.add_modeled_node(node_sensor)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        groups = complete_groups[DummySystem][sp]
        self.assertEqual(
            len(groups),
            0,
            "backward seed at every Sensor must be vetoed by the "
            "NoStepRule (each Sensor's parent is a forbidden Room); "
            f"got {len(groups)} unexpected complete groups",
        )

    def test_intermediate_hash_direction_disambiguation(self):
        """Forward and backward ``_SinglePath`` intermediates seeded
        from the same ``(sm_object, subject, predicate, object)``
        quadruple must have *distinct* :class:`Node` identities.

        The intermediate hash includes :attr:`Direction.sentinel` so
        the bidirectional walker can produce a forward intermediate
        and a backward intermediate from the same SM object without
        them collapsing onto each other (which would cross-pollute
        their adjacency views and corrupt the ruleset).
        """
        # Local application imports
        from twin4build.translator.translator import (
            BACKWARD,
            FORWARD,
            Predicate,
            _SinglePath,
        )
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_room = Node(cls=core.namespace.BRICK.Room)
        sp = SignaturePattern(id="hash_disambig_pattern")
        # Register the rule so the predicate is wired into the SP.
        sp.add_rule(
            PathRule(
                subject=node_ahu,
                object=node_room,
                predicate=core.namespace.BRICK.feeds,
            )
        )

        # Build a fresh _SinglePath whose endpoints share node_ahu /
        # node_room with the path_rule above.  Its apply() emits
        # intermediates whose hash takes the direction sentinel.
        single_path = _SinglePath(
            subject=node_ahu,
            object=node_room,
            predicate=Predicate(core.namespace.BRICK.feeds),
        )
        single_path.subject.set_signature_pattern(sp)
        single_path.object.set_signature_pattern(sp)

        # Pick any SM instance to play the role of the matched far node.
        ahu_sm = next(
            iter(
                self.semantic_model.get_instances_of_type(core.namespace.BRICK.AHU)
            )
        )

        # Forward apply -- emits a forward intermediate, then resets
        # the per-direction first_entry so the test does not stick.
        single_path._first_entry = {FORWARD: True, BACKWARD: True}
        pairs_fwd, _, _, _ = single_path.apply(
            ahu_sm,
            [ahu_sm],
            ruleset={},
            candidate_maps=[],
            direction=FORWARD,
        )
        single_path._first_entry = {FORWARD: True, BACKWARD: True}
        pairs_back, _, _, _ = single_path.apply(
            ahu_sm,
            [ahu_sm],
            ruleset={},
            candidate_maps=[],
            direction=BACKWARD,
        )

        self.assertEqual(len(pairs_fwd), 1, "forward apply should emit one pair")
        self.assertEqual(len(pairs_back), 1, "backward apply should emit one pair")

        intermediate_fwd = pairs_fwd[0][2]
        intermediate_back = pairs_back[0][2]

        self.assertNotEqual(
            intermediate_fwd,
            intermediate_back,
            "forward and backward intermediates seeded from the same SM "
            "object must be distinct Node identities (disambiguated by "
            "Direction.sentinel in the hash)",
        )
        self.assertNotEqual(
            hash(intermediate_fwd),
            hash(intermediate_back),
            "forward and backward intermediates must have distinct hash "
            "values to avoid colliding in the ruleset",
        )

        # Forward intermediate carries forward adjacency only; backward
        # intermediate carries backward adjacency only.  Confirms
        # ``Direction.set_intermediate_far_edge`` wired the right view.
        self.assertTrue(intermediate_fwd.predicate_object_pairs)
        self.assertFalse(intermediate_fwd.predicate_subject_pairs)
        self.assertFalse(intermediate_back.predicate_object_pairs)
        self.assertTrue(intermediate_back.predicate_subject_pairs)

    def test_backward_walk_optional_steprule(self):
        """Backward walk into an :class:`OptionalRule` wrapping a
        :class:`StepRule` produces matches when the optional edge is
        present in the SM and does not prune when absent.
        """
        # Local application imports
        import twin4build.core as core

        node_ahu = Node(cls=core.namespace.BRICK.AHU)
        node_damper = Node(cls=core.namespace.BRICK.Damper)

        sp = SignaturePattern(id="backward_optional_steprule_pattern")
        sp.add_rule(
            OptionalRule(
                subject=node_ahu,
                object=node_damper,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_modeled_node(node_damper)

        class DummySystem(core.System):
            pass

        DummySystem.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummySystem], semantic_model=self.semantic_model
        )

        groups = complete_groups[DummySystem][sp]
        # All four BRICK.Damper instances in the setUp are valid
        # candidates for ``node_damper``; two have an incoming
        # AHU --feeds--> edge, two do not.  OptionalRule does not prune
        # the latter, so all four produce complete groups.
        self.assertEqual(
            len(groups),
            4,
            "expected one complete group per BRICK.Damper in the setUp "
            "(OptionalRule does not prune the AHU-less dampers); got "
            f"{len(groups)}",
        )

    def test_inverse_index_sm_symmetry(self):
        """``SemanticInstance.get_predicate_subject_pairs`` is the symmetric
        counterpart of ``get_predicate_object_pairs``: every direct triple
        ``(s, p, o)`` in the instance graph must produce ``o`` in
        ``s.get_predicate_object_pairs()[p]`` *and* ``s`` in
        ``o.get_predicate_subject_pairs()[p]``.

        Direction reversal here is purely a graph-traversal mechanic; it
        has nothing to do with ``owl:inverseOf``.  This test seeds the
        BRICK fixture from :meth:`setUp` (whose predicates are not
        declared inverse / symmetric / transitive / equivalent), so the
        two views must match the direct triples exactly with no inferred
        entries.
        """
        # Local application imports
        import twin4build.core as core

        # Skip rdf:type triples -- those map class URIs which become
        # SemanticType (no inverse-edge view by design); we only assert
        # symmetry on instance-to-instance edges.
        for s, p, o in self.semantic_model.instance_graph:
            if p == core.namespace.RDF.type:
                continue

            s_inst = self.semantic_model.get_instance(s)
            o_inst = self.semantic_model.get_instance(o)
            p_obj = self.semantic_model.get_predicate(p)

            # Forward view contains the outgoing edge.
            forward = s_inst.get_predicate_object_pairs()
            self.assertIn(
                p_obj,
                forward,
                f"predicate {p} missing on outgoing view of {s}",
            )
            self.assertIn(
                o_inst,
                forward[p_obj],
                f"forward edge {s} --{p}--> {o} missing in get_predicate_object_pairs",
            )

            # Inverse view contains the incoming edge under the same
            # predicate (no owl:inverseOf required).
            inverse = o_inst.get_predicate_subject_pairs()
            self.assertIn(
                p_obj,
                inverse,
                f"predicate {p} missing on incoming view of {o}",
            )
            self.assertIn(
                s_inst,
                inverse[p_obj],
                f"inverse edge {s} --{p}--> {o} missing in get_predicate_subject_pairs",
            )

    def test_inverse_index_literal_stub(self):
        """``SemanticLiteral`` returns an empty dict from
        :meth:`get_predicate_object_pairs` (literals are RDF leaves --
        they cannot appear as the subject of a triple).  An *unattached*
        literal -- one that doesn't appear as the object of any triple
        in any model -- also returns an empty dict from
        :meth:`get_predicate_subject_pairs`.
        """
        # Local application imports
        from twin4build.model.semantic_model.semantic_model import SemanticLiteral

        sm = SemanticModel()
        lit = SemanticLiteral("hello", sm)
        self.assertEqual(lit.get_predicate_object_pairs(), {})
        # A literal that does not appear as an object of any triple in
        # the SM has no incoming edges either.
        self.assertEqual(lit.get_predicate_subject_pairs(), {})

    def test_literal_get_predicate_subject_pairs_exposes_incoming_edges(self):
        """``SemanticLiteral.get_predicate_subject_pairs`` must expose
        every triple ``(s, p, lit)`` in the instance graph as ``p ->
        [s]``.

        Regression for the cascade where SAREF
        :class:`pid_controller_signature_pattern` failed to match
        ``office_temperature_heating_controller`` -- and as a downstream
        consequence the ``office_valve_position_sensor`` /
        ``office_damper_position_sensor`` were dropped from the
        simulation model.

        Root cause: the PID pattern declares a ``StepRule`` whose
        far-end node is typed as :data:`XSD.boolean` (the
        ``SetpointController --isReverse--> "true"^^xsd:boolean``
        edge).  The bidirectional matcher walks every SP edge from
        *both* endpoints to verify it.  When the far endpoint binds
        to a :class:`SemanticLiteral`, the backward verification
        consults ``literal.get_predicate_subject_pairs()`` -- which,
        before this fix, returned the empty dict inherited from
        :class:`SemanticObject`.  The walker then pruned the edge with
        ``reason=missing-predicate``, dropping every otherwise-valid
        match that involved a literal-typed pattern node.

        This test seeds a literal with two distinct incoming triples
        (different subjects, same predicate) and asserts both
        subjects are returned under the predicate key.
        """
        # Third party imports
        from rdflib import Literal, URIRef

        # Local application imports
        import twin4build.core as core
        from twin4build.model.semantic_model.semantic_model import SemanticLiteral

        sm = SemanticModel()
        base = "http://example.org/literal_incoming#"
        ctrl_a = URIRef(base + "controller_A")
        ctrl_b = URIRef(base + "controller_B")
        is_reverse = core.namespace.S4BLDG.isReverse
        true_lit = Literal("true", datatype=core.namespace.XSD.boolean)

        sm.instance_graph.add(
            (ctrl_a, core.namespace.RDF.type, core.namespace.S4BLDG.SetpointController)
        )
        sm.instance_graph.add(
            (ctrl_b, core.namespace.RDF.type, core.namespace.S4BLDG.SetpointController)
        )
        sm.instance_graph.add((ctrl_a, is_reverse, true_lit))
        sm.instance_graph.add((ctrl_b, is_reverse, true_lit))

        lit_inst = SemanticLiteral(true_lit, sm)
        incoming = lit_inst.get_predicate_subject_pairs()

        self.assertEqual(
            len(incoming),
            1,
            f"expected exactly one predicate (isReverse) on the literal, got {list(incoming)}",
        )
        pred_obj = next(iter(incoming))
        self.assertEqual(str(pred_obj.uri), str(is_reverse))
        subjects = {str(s.uri) for s in incoming[pred_obj]}
        self.assertEqual(
            subjects,
            {str(ctrl_a), str(ctrl_b)},
            "both controllers pointing at the literal must appear in the subject list",
        )

    def test_steprule_with_literal_far_endpoint_bidirectional(self):
        """A ``StepRule`` whose far-end node binds to a literal must
        be matched correctly by the bidirectional walker.

        End-to-end regression for the PID controller cascade
        (see :meth:`test_literal_get_predicate_subject_pairs_exposes_incoming_edges`
        for the underlying root cause).  Seeds a minimal SM that
        mirrors the
        ``SetpointController --isReverse--> "true"^^xsd:boolean``
        edge from the example, and asserts the matcher produces
        exactly one complete group with the literal correctly bound
        to the boolean SP node.
        """
        # Third party imports
        from rdflib import Literal, URIRef

        # Local application imports
        import twin4build.core as core

        sm = SemanticModel()
        base = "http://example.org/steprule_literal#"
        ctrl_uri = URIRef(base + "ctrl")
        true_lit = Literal("true", datatype=core.namespace.XSD.boolean)

        sm.instance_graph.add(
            (ctrl_uri, core.namespace.RDF.type, core.namespace.S4BLDG.SetpointController)
        )
        sm.instance_graph.add(
            (ctrl_uri, core.namespace.S4BLDG.isReverse, true_lit)
        )

        ctrl_node = Node(cls=core.namespace.S4BLDG.SetpointController)
        bool_node = Node(cls=core.namespace.XSD.boolean)

        sp = SignaturePattern(id="pid_isreverse_literal_pattern")
        sp.add_rule(
            StepRule(
                subject=ctrl_node,
                object=bool_node,
                predicate=core.namespace.S4BLDG.isReverse,
            )
        )
        sp.add_modeled_node(ctrl_node)

        class DummyPIDStub(core.System):
            pass

        DummyPIDStub.sp = [sp]

        complete_groups, _ = Translator._match_patterns(
            systems_=[DummyPIDStub], semantic_model=sm
        )
        groups = complete_groups[DummyPIDStub][sp]
        self.assertEqual(
            len(groups),
            1,
            "expected exactly one complete match for the literal-far-endpoint "
            f"StepRule, got {len(groups)}",
        )
        binding = groups[0]
        self.assertEqual(str(binding[ctrl_node].uri), str(ctrl_uri))
        # The literal binding's URI is the rdflib Literal; compare by string.
        self.assertEqual(str(binding[bool_node].uri), "true")
        self.assertEqual(
            str(binding[bool_node].uri.datatype), str(core.namespace.XSD.boolean)
        )


class TestSignaturePattern(unittest.TestCase):
    def test_node_creation(self):
        """Test creating nodes for signature patterns."""
        # Local application imports
        import twin4build.core as core

        # Create a node with a single class
        node1 = Node(cls=core.namespace.S4BLDG.Damper)
        self.assertIsNotNone(node1)

        # Create a node with multiple classes (tuple)
        node2 = Node(cls=(core.namespace.S4BLDG.Damper, core.namespace.S4BLDG.Valve))
        self.assertIsNotNone(node2)

    def test_signature_pattern_creation(self):
        """Test creating a basic signature pattern."""
        # Local application imports
        # Create signature pattern
        sp = SignaturePattern()

        self.assertIsNotNone(sp)

    def test_exact_rule(self):
        """Test creating StepRule rules."""
        # Local application imports
        import twin4build.core as core

        node1 = Node(cls=core.namespace.S4BLDG.Damper)
        node2 = Node(cls=core.namespace.S4BLDG.Controller)

        # Create an StepRule rule
        rule = StepRule(
            subject=node1, object=node2, predicate=core.namespace.SAREF.controls
        )

        self.assertIsNotNone(rule)
        self.assertEqual(rule.subject, node1)
        self.assertEqual(rule.object, node2)

    def test_optional_rule(self):
        """Test creating OptionalRule rules."""
        # Local application imports
        import twin4build.core as core

        node1 = Node(cls=core.namespace.S4BLDG.Damper)
        node2 = Node(cls=core.namespace.SAREF.Property)

        # Create an OptionalRule rule
        rule = OptionalRule(
            subject=node1, object=node2, predicate=core.namespace.SAREF.hasProperty
        )

        self.assertIsNotNone(rule)
        self.assertEqual(rule.subject, node1)
        self.assertEqual(rule.object, node2)

    def test_add_triple_to_pattern(self):
        """Test adding triples to signature patterns."""
        # Local application imports
        import twin4build.core as core

        damper_node = Node(cls=core.namespace.S4BLDG.Damper)
        controller_node = Node(cls=core.namespace.S4BLDG.Controller)

        sp = SignaturePattern()

        # Add a triple with an StepRule rule
        sp.add_rule(
            StepRule(
                subject=controller_node,
                object=damper_node,
                predicate=core.namespace.SAREF.controls,
            )
        )

        # If no exception, test passed
        self.assertTrue(True)

    def test_add_input_to_pattern(self):
        """Test adding inputs to signature patterns."""
        # Local application imports
        import twin4build.core as core

        controller_node = Node(cls=core.namespace.S4BLDG.Controller)

        sp = SignaturePattern()

        # Add an input
        sp.add_input("damperPosition", controller_node, "inputSignal")

        # If no exception, test passed
        self.assertTrue(True)

    def test_add_parameter_to_pattern(self):
        """Test adding parameters to signature patterns."""
        # Local application imports
        import twin4build.core as core

        float_node = Node(cls=core.namespace.XSD.float)

        sp = SignaturePattern()

        # Add a parameter
        sp.add_parameter("nominalAirFlowRate", float_node)

        # If no exception, test passed
        self.assertTrue(True)

    def test_add_modeled_node(self):
        """Test adding modeled nodes to signature patterns."""
        # Local application imports
        import twin4build.core as core

        damper_node = Node(cls=core.namespace.S4BLDG.Damper)

        sp = SignaturePattern()

        # Add a modeled node
        sp.add_modeled_node(damper_node)

        # If no exception, test passed
        self.assertTrue(True)

    def test_inverse_index_sp_symmetry(self):
        """``Node.predicate_subject_pairs`` is the symmetric counterpart of
        ``Node.predicate_object_pairs``: every rule added to a
        :class:`SignaturePattern` must populate both views so the
        bidirectional matcher can walk SP edges as incident edges.

        Asserts symmetry across :class:`StepRule`, :class:`PathRule`,
        :class:`AnyPathRule`, and :class:`OptionalRule` -- the four user-
        facing rule shapes that all delegate to the same
        ``predicate_object_pairs`` write site in
        :meth:`SignaturePattern.add_rule`.
        """
        # Local application imports
        import twin4build.core as core

        node_a = Node(cls=core.namespace.BRICK.AHU)
        node_b = Node(cls=core.namespace.BRICK.Damper)
        node_c = Node(cls=core.namespace.BRICK.Room)
        node_d = Node(cls=core.namespace.BRICK.Temperature_Sensor)

        sp = SignaturePattern(id="sp_symmetry_pattern")
        sp.add_rule(
            StepRule(
                subject=node_a, object=node_b, predicate=core.namespace.BRICK.feeds
            )
        )
        sp.add_rule(
            PathRule(
                subject=node_a, object=node_c, predicate=core.namespace.BRICK.feeds
            )
        )
        sp.add_rule(
            AnyPathRule(
                subject=node_b, object=node_c, predicate=core.namespace.BRICK.feeds
            )
        )
        sp.add_rule(
            OptionalRule(
                subject=node_c,
                object=node_d,
                predicate=core.namespace.BRICK.hasPoint,
            )
        )
        sp.add_modeled_node(node_a)

        # The user-added (subj, pred-uri, obj) triples that ``add_rule``
        # must materialise in both views.  Each rule type wraps the URI
        # in its own :class:`Predicate` instance, so an edge may be
        # registered under multiple distinct ``Predicate`` keys -- the
        # bidirectional contract is per-instance: under every
        # ``Predicate`` instance that stores the outgoing edge, the
        # inverse view of the object must contain the subject under the
        # *same* instance.
        expected_user_edges = [
            (node_a, core.namespace.BRICK.feeds, node_b),
            (node_a, core.namespace.BRICK.feeds, node_c),
            (node_b, core.namespace.BRICK.feeds, node_c),
            (node_c, core.namespace.BRICK.hasPoint, node_d),
        ]

        for subj, pred_uri, obj in expected_user_edges:
            forward_preds = [
                p
                for p, objs in subj.predicate_object_pairs.items()
                if pred_uri in p.preds and obj in objs
            ]
            self.assertGreater(
                len(forward_preds),
                0,
                f"forward edge {subj.id} --{pred_uri}--> {obj.id} missing "
                "in predicate_object_pairs",
            )
            for pred in forward_preds:
                self.assertIn(
                    pred,
                    obj.predicate_subject_pairs,
                    f"predicate-instance for {pred_uri} missing on "
                    f"incoming view of {obj.id}",
                )
                self.assertIn(
                    subj,
                    obj.predicate_subject_pairs[pred],
                    f"inverse edge {subj.id} --{pred_uri}--> {obj.id} "
                    "missing in predicate_subject_pairs under the same "
                    "Predicate instance",
                )

    def test_weakly_connected_components_single_component(self):
        """A pattern wired into a single chain is one WCC; member order
        within the component matches SP node-registration order."""
        # Local application imports
        import twin4build.core as core

        node_a = Node(cls=core.namespace.BRICK.AHU)
        node_b = Node(cls=core.namespace.BRICK.Damper)
        node_c = Node(cls=core.namespace.BRICK.Room)

        sp = SignaturePattern(id="wcc_single_component_pattern")
        sp.add_rule(
            StepRule(
                subject=node_a, object=node_b, predicate=core.namespace.BRICK.feeds
            )
        )
        sp.add_rule(
            StepRule(
                subject=node_b, object=node_c, predicate=core.namespace.BRICK.feeds
            )
        )

        wccs = sp.weakly_connected_components()
        self.assertEqual(len(wccs), 1, f"expected 1 WCC, got {len(wccs)}")
        self.assertEqual(set(wccs[0]), {node_a, node_b, node_c})

    def test_weakly_connected_components_two_disconnected_components(self):
        """Two disjoint subgraphs in the same SignaturePattern produce
        two WCCs; the symmetric :attr:`Node.predicate_subject_pairs`
        edges are walked, so a node only reachable backward still ends
        up in the same WCC as its predecessor.
        """
        # Local application imports
        import twin4build.core as core

        node_a1 = Node(cls=core.namespace.BRICK.AHU)
        node_b1 = Node(cls=core.namespace.BRICK.Damper)
        node_a2 = Node(cls=core.namespace.BRICK.AHU)
        node_b2 = Node(cls=core.namespace.BRICK.Damper)

        sp = SignaturePattern(id="wcc_two_components_pattern")
        sp.add_rule(
            StepRule(
                subject=node_a1,
                object=node_b1,
                predicate=core.namespace.BRICK.feeds,
            )
        )
        sp.add_rule(
            StepRule(
                subject=node_a2,
                object=node_b2,
                predicate=core.namespace.BRICK.feeds,
            )
        )

        wccs = sp.weakly_connected_components()
        self.assertEqual(
            len(wccs), 2, f"expected 2 WCCs (disjoint subgraphs), got {len(wccs)}"
        )
        component_sets = [set(c) for c in wccs]
        self.assertIn({node_a1, node_b1}, component_sets)
        self.assertIn({node_a2, node_b2}, component_sets)

    def test_inverse_index_sp_initial_state(self):
        """A freshly-constructed :class:`Node` exposes both views as empty
        dicts (rather than ``None`` or missing attributes), so the
        bidirectional matcher can iterate either side without guards."""
        # Local application imports
        import twin4build.core as core

        n = Node(cls=core.namespace.BRICK.AHU)
        self.assertEqual(n.predicate_object_pairs, {})
        self.assertEqual(n.predicate_subject_pairs, {})


if __name__ == "__main__":
    unittest.main()
