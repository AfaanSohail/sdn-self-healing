"""Redundant, deterministic switch meshes; two hosts per switch."""
import networkx as nx
from dataclasses import dataclass
from config import SET

@dataclass
class Flow:
    fid: int
    src: int
    dst: int
    demand: float
    priority: int = 1


def edge(a, b):
    return tuple(sorted((a, b)))


def build_topology(n):
    assert n in (12, 20)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in ((i + 1) % n, (i + (n // 4)) % n):
            if i != j:
                g.add_edge(i, j, capacity=SET.nominal_capacity,
                           delay=SET.nominal_link_delay_ms)
    flows = [Flow(i, i, i + n//2, 9.0 + i * 2) for i in range(3)]
    # A concrete host attachment schedule: h(2*i),h(2*i+1) on switch i.
    host_links = {f'h{2*i+j}': i for i in range(n) for j in (0, 1)}
    return g, flows, host_links


def candidate_paths(graph, src, dst, weight=None, limit=3):
    if not nx.has_path(graph, src, dst):
        return []
    import itertools
    paths = list(itertools.islice(nx.shortest_simple_paths(graph, src, dst, weight=weight),
                                   limit))
    if weight is None:
        paths.sort(key=lambda p: (len(p), p))
    return paths[:limit]


def path_edges(path):
    return [edge(a, b) for a, b in zip(path, path[1:])]
