r"""
Graph algorithms for finding cycles and strongly connected components.

Mathematical Formulation:

1. Simple Cycles:
   A simple cycle in a directed graph :math:`G = (V, E)` is a path :math:`v_1, v_2, ..., v_k` where:
   - :math:`v_1 = v_k`
   - All other vertices are distinct
   - :math:`(v_i, v_{i+1}) \in E \forall i \in \{1, ..., k-1\}`

2. Strongly Connected Components (SCC):
   A strongly connected component of a directed graph :math:`G = (V, E)` is a maximal subset :math:`C \subseteq V` where:
        - For any two vertices :math:`u, v \in C`, there exists a path from :math:`u` to :math:`v`
        - For any two vertices :math:`u \in C, v \notin C`, either:
            - There is no path from :math:`u` to :math:`v`, or
            - There is no path from :math:`v` to :math:`u`

3. Tarjan's Algorithm:
   For each vertex :math:`v \in V`, we maintain:
        - :math:`index[v]`: The order in which :math:`v` was discovered
        - :math:`lowlink[v]`: The smallest index of any vertex reachable from :math:`v` through a back edge

   A vertex :math:`v` is the root of an SCC if and only if:

   .. math::

      lowlink[v] = index[v]
"""

# Standard library imports
from collections import defaultdict


def simple_cycles(G):
    # Yield every elementary cycle in python graph G exactly once
    # Expects a dictionary mapping from vertices to iterables of vertices
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()

    G = {v: set(nbrs) for (v, nbrs) in G.items()}  # make a copy of the graph
    sccs = strongly_connected_components(G)
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path = [startnode]
        blocked = set()
        closed = set()
        blocked.add(startnode)
        B = defaultdict(set)
        stack = [(startnode, list(G[startnode]))]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(G[nextnode])))
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            if not nbrs:
                if thisnode in closed:
                    _unblock(thisnode, blocked, B)
                else:
                    for nbr in G[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
                path.pop()
        remove_node(G, startnode)
        H = subgraph(G, set(scc))
        sccs.extend(strongly_connected_components(H))


def strongly_connected_components(graph):
    # Tarjan's algorithm for finding SCC's
    # Robert Tarjan. "Depth-first search and linear graph algorithms." SIAM journal on computing. 1972.
    # Code by Dries Verdegem, November 2012
    # Downloaded from http://www.logarithmic.net/pfh/blog/01208083168

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    result = []

    def _strong_connect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        successors = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node], index[successor])

        if lowlink[node] == index[node]:
            connected_component = []

            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node:
                    break
            result.append(connected_component[:])

    for node in graph:
        if node not in index:
            _strong_connect(node)

    return result


def remove_node(G, target):
    # Completely remove a node from the graph
    # Expects values of G to be sets
    del G[target]
    for nbrs in G.values():
        nbrs.discard(target)


def subgraph(G, vertices):
    # Get the subgraph of G induced by set vertices
    # Expects values of G to be sets
    return {v: G[v] & vertices for v in vertices}
