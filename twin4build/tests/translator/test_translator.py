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
    OptionalRule,
    SignaturePattern,
    PathRule,
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
        """``SemanticLiteral`` (and any other non-instance ``SemanticObject``)
        returns an empty dict from ``get_predicate_subject_pairs`` -- the
        symmetric counterpart of the existing literal stub on
        ``get_predicate_object_pairs``.
        """
        # Local application imports
        from twin4build.model.semantic_model.semantic_model import SemanticLiteral

        sm = SemanticModel()
        lit = SemanticLiteral("hello", sm)
        self.assertEqual(lit.get_predicate_object_pairs(), {})
        self.assertEqual(lit.get_predicate_subject_pairs(), {})


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
