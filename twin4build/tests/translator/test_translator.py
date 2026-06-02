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
