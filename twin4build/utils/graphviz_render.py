"""Render DOT graphs with pip Graphviz (pygraphviz 2.0+) or system binaries.

Twin4Build historically laid out semantic-model drawings with this pipeline::

    ccomps -x                  # split weakly connected components
    dot                        # hierarchical layout of each component
    gvpack -array3             # pack components into a 3-column array
    neato -n2 -Gsize=10! -Gdpi={dpi} -Grankdir=RL

``pygraphviz`` 2.0 wheels bundle the Graphviz libraries and layout plugins
but not the ``dot``, ``neato``, or ``gvpack`` executables. Layout is done
in-process:

* ``pack=true`` / ``packmode=array_3`` on ``AGraph.layout(prog="dot")`` is
  Graphviz's built-in equivalent of ``ccomps | dot | gvpack -array3``.
* ``AGraph.draw()`` after layout uses ``nop2``, which is ``neato -n2``
  (use existing node and edge positions; do not re-layout).

The system-binary path is kept as a fallback for environments that already
have a complete Graphviz install on ``PATH``.
"""

# Standard library imports
import os
import shutil
import subprocess
import warnings
from typing import Iterable, List, Optional, Sequence

# Local application imports
from twin4build.utils.logger import LOGGER

_SYSTEM_TOOLS = ("ccomps", "dot", "gvpack", "neato")

# Render-time flags historically passed to ``neato -n2``.
_PACKED_DRAW_ARGS = "-Gsize=10! -Grankdir=RL"
_PACK_MODE = "array_3"

DRAWING_UNAVAILABLE_HINT = (
    "Graphviz drawing is unavailable. It is provided by the pygraphviz "
    "wheel; reinstall twin4build or pip install 'pygraphviz>=2.0.1'. "
    "Set draw_semantic_model=False or draw_simulation_model=False to skip "
    "drawing."
)


class DrawingUnavailableError(RuntimeError):
    """Neither pygraphviz nor a complete system Graphviz install is usable."""


def pygraphviz_available() -> bool:
    """Return True if the pygraphviz package can be imported."""
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        return False
    return True


def system_graphviz_available() -> bool:
    """Return True if the four Graphviz CLI tools used by Twin4Build are on PATH."""
    return all(shutil.which(name) is not None for name in _SYSTEM_TOOLS)


def drawing_available() -> bool:
    """Return True if a drawing backend can run."""
    return pygraphviz_available() or system_graphviz_available()


def drawing_backend() -> Optional[str]:
    """Preferred backend name: ``"pygraphviz"``, ``"system"``, or ``None``."""
    if pygraphviz_available():
        return "pygraphviz"
    if system_graphviz_available():
        return "system"
    return None


def render_dot_graph(
    dot_filename: str,
    output_path: str,
    format: str = "svg",
    dpi: int = 2000,
    generate_subgraphs: bool = False,
    temp_dir: Optional[str] = None,
    subgraph_dir: Optional[str] = None,
) -> str:
    """Lay out ``dot_filename`` and write ``output_path``.

    Parameters
    ----------
    dot_filename:
        Path to a DOT file (typically written by pydotplus).
    output_path:
        Destination image path, including the file extension.
    format:
        Graphviz output format (``svg`` or ``png``).
    dpi:
        DPI used at render time (matches the historical ``neato -Gdpi`` flag).
    generate_subgraphs:
        If True, also render each weakly connected component as its own image
        in ``subgraph_dir``.
    temp_dir:
        Directory for packed/joined intermediate DOT files (system fallback).
    subgraph_dir:
        Directory for per-component DOT files and optional subgraph images.

    Returns
    -------
    str
        The backend that produced the output (``"pygraphviz"`` or ``"system"``).
    """
    backend = drawing_backend()
    if backend is None:
        raise DrawingUnavailableError(DRAWING_UNAVAILABLE_HINT)

    if generate_subgraphs and not subgraph_dir:
        raise ValueError("generate_subgraphs=True requires subgraph_dir")

    if backend == "pygraphviz":
        try:
            _render_with_pygraphviz(
                dot_filename=dot_filename,
                output_path=output_path,
                format=format,
                dpi=dpi,
                generate_subgraphs=generate_subgraphs,
                temp_dir=temp_dir,
                subgraph_dir=subgraph_dir,
            )
            return "pygraphviz"
        except DrawingUnavailableError:
            raise
        except Exception as exc:
            if not system_graphviz_available():
                raise
            warnings.warn(
                f"pygraphviz drawing failed ({exc}); falling back to system Graphviz.",
                RuntimeWarning,
                stacklevel=2,
            )
            LOGGER.warning(
                "pygraphviz drawing failed (%s); falling back to system Graphviz.",
                exc,
            )

    _render_with_system_tools(
        dot_filename=dot_filename,
        output_path=output_path,
        format=format,
        dpi=dpi,
        generate_subgraphs=generate_subgraphs,
        temp_dir=temp_dir,
        subgraph_dir=subgraph_dir,
    )
    return "system"


def weakly_connected_components(node_names: Sequence[str], edges: Iterable) -> List[List[str]]:
    """Return weakly connected components of a (possibly directed) graph.

    ``edges`` is an iterable of ``(u, v)`` name pairs. Isolated nodes listed
    in ``node_names`` form their own components. This is the Python equivalent
    of ``ccomps -x``.
    """
    adj = {name: set() for name in node_names}
    for edge in edges:
        u, v = edge[0], edge[1]
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    seen = set()
    components: List[List[str]] = []
    for name in adj:
        if name in seen:
            continue
        stack = [name]
        seen.add(name)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adj[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _clear_dir(dirname: str) -> None:
    if not dirname or not os.path.isdir(dirname):
        return
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def _run_quiet(args: List[str]) -> None:
    subprocess.run(args=args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _node_name(node) -> str:
    return node.name if hasattr(node, "name") else str(node)


def _edge_names(edge) -> tuple:
    return (_node_name(edge[0]), _node_name(edge[1]))


def _copy_component(source, keep_names: Sequence[str]):
    """Independent AGraph containing only ``keep_names``, with attributes copied.

    Reloading from DOT preserves HTML-like labels that rdf2dot emits.
    """
    keep = set(keep_names)
    component = source.__class__(string=source.to_string())
    for node in list(component.nodes()):
        if _node_name(node) not in keep:
            component.delete_node(node)
    return component


def _render_with_pygraphviz(
    dot_filename: str,
    output_path: str,
    format: str,
    dpi: int,
    generate_subgraphs: bool,
    temp_dir: Optional[str],
    subgraph_dir: Optional[str],
) -> None:
    # Third party imports
    import pygraphviz as pgv

    graph = pgv.AGraph(dot_filename)
    node_names = [_node_name(node) for node in graph.nodes()]
    edges = [_edge_names(edge) for edge in graph.edges()]
    components = weakly_connected_components(node_names, edges)

    if generate_subgraphs and subgraph_dir:
        os.makedirs(subgraph_dir, exist_ok=True)
        _clear_dir(subgraph_dir)
        for index, names in enumerate(components):
            suffix = "" if index == 0 else f"_{index}"
            component = _copy_component(graph, names)
            component_dot = os.path.join(
                subgraph_dir, f"object_graph_ccomps{suffix}.dot"
            )
            component.write(component_dot)
            # Historical ``dot -T{format}`` on the un-packed component: layout
            # and render in one step, without pack / size / dpi / rankdir.
            component.layout(prog="dot")
            component.draw(
                os.path.join(subgraph_dir, f"object_graph_ccomps{suffix}.{format}"),
                format=format,
            )
            laid_out_dot = os.path.join(
                subgraph_dir, f"object_graph_dot{suffix}.dot"
            )
            component.write(laid_out_dot)

    # ccomps | dot | gvpack -array3
    graph.graph_attr["pack"] = "true"
    graph.graph_attr["packmode"] = _PACK_MODE
    graph.layout(prog="dot")

    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        graph.write(os.path.join(temp_dir, "object_graph_gvpack.dot"))

    # Positions are already packed. Clear pack so nop2 (neato -n2) does not
    # pack again while applying the historical render-time graph flags.
    graph.graph_attr["pack"] = "false"
    if "packmode" in graph.graph_attr:
        del graph.graph_attr["packmode"]

    # neato -n2 -Gsize=10! -Gdpi={dpi} -Grankdir=RL
    draw_args = f"{_PACKED_DRAW_ARGS} -Gdpi={dpi}"
    graph.draw(output_path, format=format, args=draw_args)


def _render_with_system_tools(
    dot_filename: str,
    output_path: str,
    format: str,
    dpi: int,
    generate_subgraphs: bool,
    temp_dir: Optional[str],
    subgraph_dir: Optional[str],
) -> None:
    """Exact historical CLI pipeline: ccomps | dot | gvpack | neato -n2."""
    if subgraph_dir is None:
        subgraph_dir = os.path.join(os.path.dirname(dot_filename), "ccomps")
    if temp_dir is None:
        temp_dir = os.path.dirname(dot_filename)

    os.makedirs(subgraph_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    _clear_dir(subgraph_dir)

    ccomps = shutil.which("ccomps")
    dot = shutil.which("dot")
    gvpack = shutil.which("gvpack")
    neato = shutil.which("neato")
    missing = [
        name
        for name, path in (
            ("ccomps", ccomps),
            ("dot", dot),
            ("gvpack", gvpack),
            ("neato", neato),
        )
        if path is None
    ]
    if missing:
        raise DrawingUnavailableError(
            "Missing Graphviz executable(s): "
            + ", ".join(missing)
            + ". "
            + DRAWING_UNAVAILABLE_HINT
        )

    ccomps_prefix = os.path.join(subgraph_dir, "object_graph_ccomps.dot")
    _run_quiet([ccomps, "-x", f"-o{ccomps_prefix}", dot_filename])

    laid_out = []
    for filename in os.listdir(subgraph_dir):
        file_path = os.path.join(subgraph_dir, filename)
        if not os.path.isfile(file_path) or not filename.endswith(".dot"):
            continue
        if "ccomps" not in filename:
            continue
        laid_out_dot = os.path.join(
            subgraph_dir, filename.replace("ccomps", "dot")
        )
        _run_quiet([dot, "-q", f"-o{laid_out_dot}", file_path])
        if generate_subgraphs:
            image_path = file_path.replace(".dot", f".{format}")
            _run_quiet([dot, f"-T{format}", "-q", f"-o{image_path}", file_path])
        laid_out.append(laid_out_dot)

    joined_dot = os.path.join(temp_dir, "object_graph_ccomps_joined.dot")
    with open(joined_dot, "wb") as dest:
        for path in laid_out:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, dest)

    packed_dot = os.path.join(temp_dir, "object_graph_gvpack.dot")
    _run_quiet([gvpack, "-array3", f"-o{packed_dot}", joined_dot])
    _run_quiet(
        [
            neato,
            f"-T{format}",
            "-n2",
            "-Gsize=10!",
            f"-Gdpi={dpi}",
            "-Grankdir=RL",
            "-q",
            f"-o{output_path}",
            packed_dot,
        ]
    )
