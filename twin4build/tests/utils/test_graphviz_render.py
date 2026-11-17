# Standard library imports
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

# Local application imports
import twin4build
from twin4build.utils.graphviz_render import (
    DRAWING_UNAVAILABLE_HINT,
    DrawingUnavailableError,
    drawing_available,
    drawing_backend,
    pygraphviz_available,
    render_dot_graph,
    weakly_connected_components,
)

twin4build._IS_TESTING = True

_TWO_COMPONENT_DOT = """
digraph G {
  a [label="AlphaNode"];
  b [label="BetaNode"];
  c [label="GammaNode"];
  d [label="DeltaNode"];
  a -> b;
  c -> d;
}
"""

_HTML_LABEL_DOT = """
digraph {
  "http://example.org/a" [label=<<TABLE BORDER="2" CELLBORDER="0"><TR><TD>HeaderA</TD></TR><TR><TD>http://example.org/a</TD></TR></TABLE>>];
  "http://example.org/b" [label=<<TABLE BORDER="2" CELLBORDER="0"><TR><TD>HeaderB</TD></TR><TR><TD>http://example.org/b</TD></TR></TABLE>>];
  "http://example.org/a" -> "http://example.org/b";
}
"""


class TestWeaklyConnectedComponents(unittest.TestCase):
    def test_two_directed_components(self):
        components = weakly_connected_components(
            ["a", "b", "c", "d"],
            [("a", "b"), ("c", "d")],
        )
        as_sets = {frozenset(c) for c in components}
        self.assertEqual(as_sets, {frozenset({"a", "b"}), frozenset({"c", "d"})})

    def test_isolated_nodes(self):
        components = weakly_connected_components(["a", "b"], [])
        as_sets = {frozenset(c) for c in components}
        self.assertEqual(as_sets, {frozenset({"a"}), frozenset({"b"})})

    def test_weak_connectivity_ignores_direction(self):
        components = weakly_connected_components(
            ["a", "b", "c"],
            [("a", "b"), ("c", "b")],
        )
        self.assertEqual(len(components), 1)
        self.assertEqual(set(components[0]), {"a", "b", "c"})


class TestRenderDotGraph(unittest.TestCase):
    def setUp(self):
        if not drawing_available():
            self.skipTest("Graphviz drawing backend is not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_dot(self, contents, name="graph.dot"):
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(contents)
        return path

    def test_backend_prefers_pygraphviz_when_installed(self):
        if pygraphviz_available():
            self.assertEqual(drawing_backend(), "pygraphviz")

    def test_packed_two_component_svg_keeps_both_components(self):
        """ccomps | dot | gvpack -array3 | neato -n2 equivalent must keep both islands."""
        dot_path = self._write_dot(_TWO_COMPONENT_DOT)
        output = os.path.join(self.temp_dir, "packed.svg")
        temp_dir = os.path.join(self.temp_dir, "temp")
        os.makedirs(temp_dir)
        backend = render_dot_graph(
            dot_filename=dot_path,
            output_path=output,
            format="svg",
            dpi=100,
            temp_dir=temp_dir,
        )
        self.assertIn(backend, ("pygraphviz", "system"))
        self.assertTrue(os.path.isfile(output))
        with open(output, "r", encoding="utf-8") as handle:
            svg = handle.read()
        self.assertGreater(len(svg), 0)
        self.assertIn("AlphaNode", svg)
        self.assertIn("BetaNode", svg)
        self.assertIn("GammaNode", svg)
        self.assertIn("DeltaNode", svg)
        packed_dot = os.path.join(temp_dir, "object_graph_gvpack.dot")
        self.assertTrue(os.path.isfile(packed_dot))

    def test_html_labels_survive_layout(self):
        dot_path = self._write_dot(_HTML_LABEL_DOT, name="html.dot")
        output = os.path.join(self.temp_dir, "html.svg")
        render_dot_graph(
            dot_filename=dot_path,
            output_path=output,
            format="svg",
            dpi=100,
            temp_dir=self.temp_dir,
        )
        with open(output, "r", encoding="utf-8") as handle:
            svg = handle.read()
        self.assertIn("HeaderA", svg)
        self.assertIn("HeaderB", svg)

    def test_generate_subgraphs_writes_per_component_images(self):
        dot_path = self._write_dot(_TWO_COMPONENT_DOT)
        output = os.path.join(self.temp_dir, "packed.svg")
        subgraph_dir = os.path.join(self.temp_dir, "ccomps")
        os.makedirs(subgraph_dir)
        render_dot_graph(
            dot_filename=dot_path,
            output_path=output,
            format="svg",
            dpi=100,
            generate_subgraphs=True,
            temp_dir=self.temp_dir,
            subgraph_dir=subgraph_dir,
        )
        component_svgs = [
            name
            for name in os.listdir(subgraph_dir)
            if name.startswith("object_graph_ccomps") and name.endswith(".svg")
        ]
        self.assertGreaterEqual(len(component_svgs), 2)
        self.assertTrue(os.path.isfile(output))

    def test_png_render_time_dpi(self):
        dot_path = self._write_dot(_TWO_COMPONENT_DOT)
        output = os.path.join(self.temp_dir, "packed.png")
        render_dot_graph(
            dot_filename=dot_path,
            output_path=output,
            format="png",
            dpi=72,
            temp_dir=self.temp_dir,
        )
        self.assertTrue(os.path.isfile(output))
        self.assertGreater(os.path.getsize(output), 0)


class TestDrawingUnavailable(unittest.TestCase):
    def test_hint_mentions_skip_flags(self):
        self.assertIn("draw_semantic_model=False", DRAWING_UNAVAILABLE_HINT)

    def test_render_raises_when_no_backend(self):
        with patch(
            "twin4build.utils.graphviz_render.drawing_backend", return_value=None
        ):
            with self.assertRaises(DrawingUnavailableError):
                render_dot_graph(
                    dot_filename="missing.dot",
                    output_path="out.svg",
                )


if __name__ == "__main__":
    unittest.main()
