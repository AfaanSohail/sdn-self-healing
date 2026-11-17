"""Stochastic seeded service KPI surrogate. NOT Mininet or a physical network."""
import collections
import random
import networkx as nx
import numpy as np
from topology import build_topology, candidate_paths, path_edges, edge
from config import SET, LOAD_BASE

class NetworkEpisode:
    def __init__(self, topology_size, scenario, load, seed):
        self.n, self.scenario, self.load, self.seed = topology_size, scenario, load, seed
        self.graph, self.flows, self.host_links = build_topology(topology_size)
        self.edges = sorted(edge(*e) for e in self.graph.edges)
        self.rng = random.Random(seed + 31 * topology_size)
        self.background = {e: LOAD_BASE[load] + self.rng.uniform(-3, 2) for e in self.edges}
        self.original = {f.fid: candidate_paths(self.graph, f.src, f.dst, limit=1)[0]
                         for f in self.flows}
        self.routes = {k: list(v) for k, v in self.original.items()}
        self.strength = self.rng.uniform(.45, 1.35)
        self.delay_base = {e: self.rng.uniform(1.,3.) for e in self.edges}
        self.noise = np.random.default_rng(seed + topology_size*100).normal(size=(180,len(self.edges),4))
        self.drop = np.random.default_rng(seed+88).random((180,len(self.edges))) < .035
        self.drop_runs = {}
        self.counters = {e:0. for e in self.edges}
        self.last_counter = {}
        self.measurements = {}
        self.last_actual = None
        self.pending_harm = None
        self.diag_scores = {}
        self.raw_sample = []
        self.previous = {}
        self.last_change = {f.fid: -999 for f in self.flows}
        self.timeseries = collections.deque(maxlen=SET.history)
        self.time = -1
        self.acked = 0
        self.route_changes = 0
        self.rollbacks = 0
        self.harmful = 0
        self._targets = self._make_targets()
        self.first_stable = None
        self.streak = 0

    def _make_targets(self):
        primary = list(path_edges(self.original[0]))
        second = list(path_edges(self.original[1]))
        other = next((e for e in second if e not in primary), None)
        if other is None:
            other = next(e for e in self.edges if e not in primary)
        # Choose another edge near the original route where appropriate.
        return {'main': self.rng.choice(primary), 'second': other,
                'second_main': self.rng.choice(primary),
                'switch': self.rng.choice(self.original[0][1:]),
                'source': self.flows[0].src}

    def fault_map(self, t):
        if t < SET.inject_at or t >= SET.clear_at:
            return {}, set()
        sc = self.scenario
        bad = {}
        down_switches = set()
        main, second = self._targets['main'], self._targets['second']
        if sc in ('outage', 'concurrent', 'cascading', 'correlated'):
            bad[main] = 'outage'
        elif sc in ('congestion', 'loss', 'delay'):
            bad[main] = sc
        if sc == 'concurrent':
            bad[second] = 'loss'
        if sc == 'correlated':
            bad[second] = 'outage'
        if sc == 'cascading' and t >= 70:
            bad[second] = 'congestion'
        if sc == 'switch':
            down_switches.add(self._targets['switch'])
            for e in self.graph.edges(self._targets['switch']):
                bad[edge(*e)] = 'outage'
        if sc == 'infeasible':
            for e in self.graph.edges(self._targets['source']):
                bad[edge(*e)] = 'outage'
        return bad, down_switches

    def active_labels(self, t):
        fm, _ = self.fault_map(t)
        return sorted(set(fm.values()))

    def observed_graph(self, t):
        fm, down = self.fault_map(t)
        g = self.graph.copy()
        g.remove_nodes_from(down)
        for e, ft in fm.items():
            if ft == 'outage' and g.has_edge(*e):
                g.remove_edge(*e)
        return g

    def physical(self,t,routes=None):
        routes=self.routes if routes is None else routes
        faults, _ = self.fault_map(t)
        # Stable exogenous demand is independent of the chosen policy.
        loads = {e: self.background[e] + 4 * self.noise[t,i,0] + 1.2 * np.sin((t + e[0]*3 + e[1]) / 13)
                 for i,e in enumerate(self.edges)}
        if self.scenario=='congestion' and 50<=t<60:
            loads[self._targets['main']] += 2.5*self.strength*(t-50)
        if self.scenario == 'burst' and 60 <= t < 75:
            loads[self._targets['main']] += 4.0
        for f in self.flows:
            for e in path_edges(routes[f.fid]):
                loads[e] += f.demand
        links = {}
        for e in self.edges:
            category = faults.get(e)
            util = max(0, loads[e] / SET.nominal_capacity)
            down = category == 'outage'
            los = .001 + max(0., util - .95) * .50
            delay = self.delay_base[e] + max(0., util - .8) * 8
            if category == 'congestion':
                ramp = min(1., max(0., (t-50)/20))
                util += .50*self.strength*ramp
                loads[e] += 50*self.strength*ramp
                los += max(0, util - 1) * .12
                delay += 16*self.strength*ramp
            elif category == 'loss':
                los += .07*self.strength
            elif category == 'delay':
                delay += 25*self.strength
            if down:
                los = 1.0
                delay = 55
            links[e] = {'util': util, 'loss': min(1., los), 'delay': delay,
                        'up': float(not down), 'age': 0., 'rate_mbps': loads[e],
                        'drops': int(los * loads[e] * 100)}
        metrics = {}
        for f in self.flows:
            pe = path_edges(routes[f.fid])
            end_loss = 1 - np.prod([1 - links[e]['loss'] for e in pe])
            rtt = 2 * sum(links[e]['delay'] for e in pe)
            share = min([min(1., 100 / max(1., links[e]['rate_mbps']))
                         for e in pe], default=0)
            goodput = f.demand * (1 - end_loss) * share
            violated = bool(end_loss > SET.loss_sla or rtt > SET.rtt_sla_ms or
                            goodput < SET.goodput_ratio_sla * f.demand)
            metrics[f.fid] = {'loss': float(end_loss), 'rtt': float(rtt),
                              'goodput': float(goodput), 'viol': int(violated)}
        return links,metrics

    def observe(self,t):
        self.time=t
        links,metrics=self.physical(t)
        actual_links={e:dict(v) for e,v in links.items()}
        # Exogenous measurement errors and missing samples, identical for paired runs.
        for i,e in enumerate(self.edges):
            v=links[e]
            self.counters[e] += max(0,v['rate_mbps'])*1e6/8
            if t==90 and i==0: self.counters[e]=0. # exercise counter resets
            missing=bool(self.drop[t,i] or (e==self._targets['main'] and 61<=t<=64 and self.seed%3==0))
            raw={'t':t,'edge':str(e),'bytes':self.counters[e],'missing':int(missing)}
            if len(self.raw_sample)<300: self.raw_sample.append(raw)
            if missing:
                age=self.drop_runs.get(e,0)+1
                self.drop_runs[e]=age
                v=dict(self.measurements.get(e,v));v['age']=float(age);v['missing']=1.
                # Keep old port status during delayed observations.
            else:
                self.drop_runs[e]=0
                prev=self.last_counter.get(e)
                # No forward information; counters only differenced across received bins.
                rate=v['rate_mbps'] if prev is None else max(0.,(self.counters[e]-prev[1])*8/1e6/max(1,t-prev[0]))
                if prev is not None and self.counters[e]<prev[1]: rate=v['rate_mbps']
                self.last_counter[e]=(t,self.counters[e])
                v=dict(v,rate_mbps=rate,age=0.,missing=0.)
                v['util']=max(0.,v['util']+.025*self.noise[t,i,1])
                v['loss']=float(np.clip(v['loss']+.001*self.noise[t,i,2],0,1))
                v['delay']=max(.1,v['delay']+.35*self.noise[t,i,3])
                self.measurements[e]=dict(v)
            links[e]=v
        # Probes are also noisy, but evaluation uses physical service values below.
        self.last_actual={'links':actual_links,'flows':{k:dict(v) for k,v in metrics.items()}}
        if self.pending_harm is not None:
            fid,old_routes=self.pending_harm
            _,before=self.physical(t,old_routes)
            harmed=any(not before[k]['viol'] and metrics[k]['viol'] for k in metrics if k!=fid)
            worsened=metrics[fid]['loss']>before[fid]['loss']+.02 or metrics[fid]['rtt']>before[fid]['rtt']+5
            if harmed or worsened:
                self.harmful+=1
                self.routes[fid]=list(old_routes[fid])
                self.rollbacks+=1;self.route_changes+=1
                self.last_change[fid]=t
            self.pending_harm=None
        obs = {'t': t, 'links': links, 'flows': metrics,
               'routes': {k: tuple(v) for k,v in self.routes.items()},
               'age': max(v['age'] for v in links.values())}
        self.timeseries.append(obs)
        return obs

    def choose_flow(self, obs):
        deficits = []
        for f in self.flows:
            m = obs['flows'][f.fid]
            d = (3 * m['viol'] + min(2., m['rtt']/SET.rtt_sla_ms) +
                 max(0, 1 - m['goodput']/f.demand)) * f.priority
            deficits.append((d, -f.fid, f.fid))
        return max(deficits)[2]

    def paths(self, fid, obs):
        f = self.flows[fid]
        g = self.graph.copy()
        g.remove_edges_from([e for e,v in obs['links'].items() if v['up']<.5])
        for a,b in g.edges:
            e = edge(a,b)
            v = obs['links'][e]
            g[a][b]['pred_delay'] = v['delay'] + 25 * v['loss'] + 6 * max(0.,v['util']-.85)
        return candidate_paths(g, f.src, f.dst, weight='pred_delay', limit=3)

    def candidate_descriptor(self, fid, path, obs):
        pe = path_edges(path)
        if not pe:
            return (0., 0., 0., 0., 0.)
        vals = [obs['links'][e] for e in pe]
        residual = min(100 - v['rate_mbps'] for v in vals)
        return (len(pe)/10, min(1., residual/100),
                max(v['loss'] for v in vals),
                min(2., 2 * sum(v['delay'] for v in vals) / 80),
                max(v['util'] for v in vals)/4 + .5*max((self.diag_scores.get(e,0.) for e in pe),default=0.))

    def get_state(self, obs, probs, class_probs, candidates, fid, ignore_diag=False):
        f = self.flows[fid]
        m = obs['flows'][fid]
        allm = list(obs['flows'].values())
        x = [np.mean([v['viol'] for v in allm]),
             min(2.,m['rtt']/50),m['loss'],min(2.,m['goodput']/f.demand),
             probs, float(self.time - self.last_change[fid] < SET.cooldown),
             min(1.,max(0.,self.time-self.last_change[fid])/10),
             min(1.,obs['age']/3),
             *([0.]*4 if ignore_diag else [class_probs.get(z,0.) for z in ('outage','congestion','loss','delay')])]
        for i in range(3):
            x.extend(self.candidate_descriptor(fid, candidates[i], obs) if i<len(candidates)
                     else (0,0,0,0,0))
        x.extend([float(tuple(self.routes[fid]) == tuple(candidates[i])) if i<len(candidates) else 0.
                  for i in range(3)])
        return np.array(x, dtype='float32')

    def current_reward(self, obs, changed=False, harmful=False, recovered_bonus=False):
        deficit = np.mean([m['viol'] for m in obs['flows'].values()])
        short = np.mean([max(0., 1 - m['goodput']/f.demand)
                         for f,m in zip(self.flows,obs['flows'].values())])
        over = np.mean([min(1.,max(0.,v['util']-.85)) for v in obs['links'].values()])
        return -4*deficit -2*short-over-.1*changed-10*harmful+2*recovered_bonus

    def update_recovery(self, obs):
        if SET.inject_at <= obs['t'] < SET.clear_at:
            good = all(not m['viol'] for m in obs['flows'].values())
            self.streak = self.streak+1 if good else 0
            if self.streak == SET.stable_period and self.first_stable is None:
                self.first_stable = obs['t'] - SET.stable_period + 1
        return self.first_stable

    def execute(self, fid, action, candidates, t):
        before_routes={k:list(v) for k,v in self.routes.items()}
        if action == 0:
            return 'noop'
        if action == 4:
            if fid not in self.previous:
                return 'no_snapshot'
            self.routes[fid] = self.previous.pop(fid)
            self.rollbacks += 1
            self.route_changes += 1
        elif 1 <= action <= 3 and action <= len(candidates):
            path = candidates[action - 1]
            if path == self.routes[fid]:
                return 'same_route'
            self.previous[fid] = list(self.routes[fid])
            self.routes[fid] = list(path)
            self.route_changes += 1
        else:
            return 'invalid_index'
        self.pending_harm=(fid,before_routes)
        self.last_change[fid] = t
        self.acked += 1
        return 'ack'
