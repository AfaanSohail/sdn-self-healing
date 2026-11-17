"""Path feasibility checks plus a retained emergency management interlock."""
import networkx as nx
from topology import path_edges
from config import SET

def check_action(env, obs, fid, action, paths, shield=True):
    if action == 0:
        return True, 'noop'
    if action == 4:
        if fid in env.previous:
            return True, 'rollback'
        return False, 'no_snapshot'
    if action < 1 or action > len(paths):
        return False, 'no_candidate'
    path = paths[action-1]
    f = env.flows[fid]
    g = env.graph.copy()
    g.remove_edges_from([e for e,v in obs['links'].items() if v['up']<.5])
    # Even A2 never disables the controller's emergency management interlock.
    if not nx.has_path(g, 0, 1):
        return False, 'management_interlock'
    if len(set(path)) != len(path) or path[0] != f.src or path[-1] != f.dst:
        return False, 'loop_or_endpoints'
    if not all(g.has_edge(a,b) for a,b in zip(path,path[1:])):
        return False, 'failed_or_missing_link'
    if not shield:
        return True, 'unshielded'
    if any(obs['links'][e]['age']>2.0 for e in path_edges(path)):
        return False, 'stale_telemetry'
    if obs['t'] - env.last_change[fid] < SET.cooldown:
        return False, 'cooldown'
    for e in path_edges(path):
        v = obs['links'][e]
        # Remove the flow's existing demand when evaluating a reused edge.
        already = f.demand if e in path_edges(env.routes[fid]) else 0.
        free = SET.nominal_capacity-v['rate_mbps']+already
        if free < SET.capacity_factor*f.demand:
            return False, 'insufficient_capacity'
    return True, 'approved'
