"""
End-to-end driver:  topology -> tasks -> schedulers (MPC + baselines)
-> CSVs + summary + plots + interactive viewer.

Run from hypatia_sim/:
    python3 -m oec_sim.run_all
"""

import csv
import os
import time

import numpy as np

from . import config as C
from . import utility
from . import topology as T
from . import tasks as TK
from .schedulers import GreedyScheduler, MPCScheduler


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'  wrote {len(rows):6d} rows -> {os.path.relpath(path)}')


def run_schedulers(topo, extra_makers=()):
    """Run every scheduler on a fresh copy of the task set. extra_makers
    is a list of (topo, tasks) -> scheduler callables appended to the
    default set (e.g. mpc-congestion, mpc-hier — see run_all.main())."""
    results = {}
    makers = ([lambda tp, tk: MPCScheduler(tp, tk),
               lambda tp, tk: GreedyScheduler(tp, tk, depth=None)]
              + [lambda tp, tk, q=q: GreedyScheduler(tp, tk, depth=q)
                 for q in C.DEPTHS]
              + list(extra_makers))
    for make in makers:
        tasks = TK.generate_tasks(topo)          # deterministic, fresh state
        sched = make(topo, tasks)
        t0 = time.time()
        hist = sched.run()
        print(f'  {sched.name:16s}  {time.time() - t0:6.1f} s  '
              f'delivered {hist["delivered_images"][-1]:12,.0f} images  '
              f'utility {hist["utility"][-1]:8.2f}')
        results[sched.name] = {
            'hist': hist, 'tasks': tasks,
            'solve_log': getattr(sched, 'solve_log', []),
            'route_rows': getattr(sched, 'route_rows', []),
            'n_route_fallbacks': getattr(sched, 'n_route_fallbacks', 0),
            'n_route_lookups': getattr(sched, 'n_route_lookups', 0),
            'n_route_exec': getattr(sched, 'n_route_exec', 0),
            'n_route_exec_fallbacks': getattr(
                sched, 'n_route_exec_fallbacks', 0)}
    return results


def write_outputs(topo, results):
    os.makedirs(C.OUT_DIR, exist_ok=True)

    wins = T.contact_windows(topo)
    _write_csv(os.path.join(C.OUT_DIR, 'contact_windows.csv'), wins,
               ['gs', 'sat', 'start_s', 'end_s', 'duration_s'])

    # per-slot topology state (aggregates; full per-link table is ~1.4M rows)
    rows = []
    intra = np.array([abs(a - b) in (1, C.SATS_PER_PLANE - 1)
                      and a // C.SATS_PER_PLANE == b // C.SATS_PER_PLANE
                      for a, b in topo.isl_pairs])
    for t in range(C.N_SLOTS):
        rows.append({
            't_s': int(topo.times[t]),
            'isl_feasible': int(topo.isl_ok[t].sum()),
            'isl_feasible_intra': int(topo.isl_ok[t][intra].sum()),
            'isl_feasible_inter': int(topo.isl_ok[t][~intra].sum()),
            'isl_dist_min_km': round(float(topo.isl_dist[t][topo.isl_ok[t]].min()), 1),
            'isl_dist_max_km': round(float(topo.isl_dist[t][topo.isl_ok[t]].max()), 1),
            **{f'sats_visible_{g["name"]}': int(topo.gsl_ok[t, :, gi].sum())
               for gi, g in enumerate(C.GROUND_STATIONS)},
        })
    _write_csv(os.path.join(C.OUT_DIR, 'topology_state.csv'), rows,
               list(rows[0].keys()))

    # tasks + per-scheduler outcomes
    any_tasks = next(iter(results.values()))['tasks']
    rows = [{'kid': k.kid, 'aoi': k.aoi, 'src_sat': k.src_sat,
             'dst_gs': C.GROUND_STATIONS[k.dst_gs]['name'],
             'arrival_s': k.arrival_slot * C.SLOT_S, 'n_images': k.n_images,
             'weight': k.weight, 'deadline_s': k.deadline_s}
            for k in any_tasks]
    _write_csv(os.path.join(C.OUT_DIR, 'tasks.csv'), rows, list(rows[0].keys()))

    for name, res in results.items():
        rows = []
        for k in res['tasks']:
            completion_s = ((k.completion_slot + 1) * C.SLOT_S
                            if k.completion_slot is not None else None)
            rows.append({
                'kid': k.kid, 'depth': k.depth,
                'arrival_s': k.arrival_slot * C.SLOT_S,
                'deadline_s': k.deadline_s,
                'delivered_images': round(k.delivered, 1),
                'delivery_fraction': round(k.delivered / k.n_images, 4),
                'utility': round(k.delivered_utility, 4),
                'first_delivery_s': ((k.first_delivery_slot + 1) * C.SLOT_S
                                     if k.first_delivery_slot is not None else None),
                'completion_s': completion_s,
                'completion_delay_s': (completion_s - k.arrival_slot * C.SLOT_S
                                       if completion_s is not None else None),
                'lateness_s': (completion_s - k.deadline_s
                               if completion_s is not None else None),
                'late_image_fraction': round(k.late_images / k.n_images, 4),
                'deadline_violated': int(
                    k.dropped or completion_s is None or completion_s > k.deadline_s),
                'dropped': int(k.dropped),
                'rejected': int(k.rejected),
                **{kk: round(vv, 6) for kk, vv in utility.components(k).items()},
            })
        _write_csv(os.path.join(C.OUT_DIR, f'task_outcomes_{name}.csv'),
                   rows, list(rows[0].keys()))
        if res.get('solve_log'):
            _write_csv(os.path.join(C.OUT_DIR, f'solve_log_{name}.csv'),
                       res['solve_log'], list(res['solve_log'][0].keys()))
        hist = res['hist']
        rows = [{'t_s': hist['t_s'][i],
                 'delivered_images': round(hist['delivered_images'][i], 1),
                 'utility': round(hist['utility'][i], 4),
                 'backlog_bits': round(hist['backlog_bits'][i]),
                 'n_active': hist['n_active'][i],
                 'n_dropped': hist['n_dropped'][i]}
                for i in range(len(hist['t_s']))]
        _write_csv(os.path.join(C.OUT_DIR, f'timeline_{name}.csv'),
                   rows, list(rows[0].keys()))
        if res.get('route_rows'):
            _write_csv(os.path.join(C.OUT_DIR, f'routes_{name}.csv'),
                       res['route_rows'], list(res['route_rows'][0].keys()))


def build_summary(topo, results):
    L = []
    L.append('=' * 74)
    L.append('OEC RQ-NAC Network Simulation — Summary')
    L.append('=' * 74)
    L.append(f'Constellation   {C.CONSTELLATION_NAME}: Walker '
             f'{C.N_SATS}/{C.N_PLANES}/{C.PHASING_F}  '
             f'({C.N_PLANES} planes x {C.SATS_PER_PLANE} sats)')
    L.append(f'Altitude        {C.ALT_KM:.0f} km   inclination {C.INCL_DEG}°  '
             f'orbital period {C.T_ORB:.0f} s')
    L.append(f'ISLs            +Grid, {len(topo.isl_pairs)} candidate links, '
             f'LOS-gated (graze > {C.LOS_GRAZE_KM:.0f} km) '
             f'+ range <= {C.ISL_MAX_KM:.0f} km')
    feas = topo.isl_ok.mean(axis=0)
    L.append(f'                feasible on average: '
             f'{topo.isl_ok.sum(axis=1).mean():.0f}/{len(topo.isl_pairs)} '
             f'({100 * feas.mean():.1f}%)')
    L.append(f'GBSs            ' + ', '.join(g['name'] for g in C.GROUND_STATIONS)
             + f'   (min elevation {C.MIN_ELEV_DEG}°)')
    L.append(f'Link rates      ISL {C.ISL_RATE_BPS / 1e6:.0f} Mbps, '
             f'GBS aggregate {C.GS_RATE_BPS / 1e6:.0f} Mbps')
    L.append(f'Simulation      {C.SIM_S} s ({C.SIM_S / 3600:.1f} h, '
             f'{C.SIM_S / C.T_ORB:.1f} orbits) at dt = {C.SLOT_S} s '
             f'-> {C.N_SLOTS} slots')
    L.append(f'Depths D        {C.DEPTHS}  payload b_q = '
             + ', '.join(f'{C.PAYLOAD_B[q]} B' for q in C.DEPTHS))
    L.append('')

    wins = T.contact_windows(topo)
    L.append('── Contact windows ──────────────────────────────────────────')
    for g in C.GROUND_STATIONS:
        gw = [w for w in wins if w['gs'] == g['name']]
        cov = topo.gsl_ok[:, :, C.GROUND_STATIONS.index(g)].any(axis=1).mean()
        L.append(f'  {g["name"]:10s}  {len(gw):3d} windows   '
                 f'avg {np.mean([w["duration_s"] for w in gw]):5.0f} s   '
                 f'>=1 sat visible {100 * cov:5.1f}% of slots')
    L.append('')

    any_tasks = next(iter(results.values()))['tasks']
    tot_img = sum(k.n_images for k in any_tasks)
    q_max = max(C.DEPTHS)
    offered16 = tot_img * C.PAYLOAD_B[q_max] * 8
    cap = C.N_GS * C.GS_RATE_BPS * C.SIM_S
    L.append('── Task load K ──────────────────────────────────────────────')
    L.append(f'  {len(any_tasks)} tasks, {tot_img:,} images total, '
             f'N_k ~ U({C.TASK_IMAGES_MIN:,}, {C.TASK_IMAGES_MAX:,})')
    L.append(f'  offered load at depth-{q_max}: {offered16 / 1e9:.1f} Gbit  vs  '
             f'total GBS capacity {cap / 1e9:.1f} Gbit  '
             f'(utilization {offered16 / cap:.2f})')
    L.append(f'  encoder pipeline: {C.ENC_IMGS_PER_S:.0f} img/s per satellite '
             f'({C.ENC_S_PER_IMG * 1e3:.2f} ms/img measured)')
    L.append('')

    L.append('── Scheduler comparison (maximize images transmitted / utility) ─')
    L.append(f'  {"scheduler":16s} {"images":>14s} {"delivery%":>10s} '
             f'{"utility":>9s} {"on-time%":>9s} {"viol%":>7s} {"dropped":>8s}')
    for name, res in results.items():
        tk = res['tasks']
        img = sum(k.delivered for k in tk)
        frac = img / tot_img
        ontime = sum(v for k in tk for t, v in k.delivery_slots.items()
                     if (t + 1) * C.SLOT_S <= k.deadline_s)
        n_viol = sum(1 for k in tk if k.dropped or k.completion_slot is None
                    or (k.completion_slot + 1) * C.SLOT_S > k.deadline_s)
        n_dropped = sum(1 for k in tk if k.dropped)
        L.append(f'  {name:16s} {img:14,.0f} {100 * frac:9.1f}% '
                 f'{res["hist"]["utility"][-1]:9.2f} {100 * ontime / tot_img:8.1f}% '
                 f'{100 * n_viol / len(tk):6.1f}% {n_dropped:8d}')
    L.append('')

    L.append('── Unified utility decomposition ────────────────────────────')
    L.append(f'  quality source: {utility.quality_source_label()}')
    L.append(f'  mode={C.UTIL_MODE}  weights  omega_Q={C.UTIL_W_QUALITY} '
             f'omega_T={C.UTIL_W_TARDY} omega_E={C.UTIL_W_COST} '
             f'omega_F={C.UTIL_W_FAIR}  segments={len(C.UTIL_COVERAGE_BREAKS)}')
    L.append(f'  s_q = ' + '  '.join(f'{q}:{v:.4f}'
                                     for q, v in sorted(utility.quality_table().items())))
    L.append(f'  {"scheduler":16s} {"total":>8s} {"quality":>8s} {"-tardy":>8s} '
             f'{"-cost":>7s} {"+fair":>7s} {"u_min":>7s} {"jain":>6s} '
             f'{"enc kJ":>8s} {"tx kJ":>7s}')
    for name, res in results.items():
        tot, terms, jain, umin = utility.run_utility(res['tasks'])
        L.append(f'  {name:16s} {tot:8.3f} {terms["quality"]:8.3f} '
                 f'{-terms["tardiness"]:8.3f} {-terms["cost"]:7.3f} '
                 f'{terms["fairness"]:7.3f} {umin:7.4f} {jain:6.4f} '
                 f'{terms["enc_energy_j"] / 1e3:8.1f} '
                 f'{terms["tx_energy_j"] / 1e3:7.2f}')
    L.append('  (Jain is a DIAGNOSTIC only -- it is a ratio of quadratics and')
    L.append('   is not MILP-representable, so u_min is what gets optimized.)')
    L.append('')

    L.append('── Routing ──────────────────────────────────────────────────')
    L.append(f'  {"scheduler":16s} {"hops":>6s} {"delay ms":>9s} '
             f'{"paths/(k,t)":>12s} {"frozen ok @exec":>16s} {"@horizon":>9s} '
             f'{"route LPs":>10s} {"route s":>8s}')
    for name, res in results.items():
        rr = res.get('route_rows') or []
        rlog = [x for x in (res.get('solve_log') or [])
                if x.get('level') == 'route']
        look = res.get('n_route_lookups', 0)
        fb = res.get('n_route_fallbacks', 0)
        if not rr and not look and not rlog:
            L.append(f'  {name:16s} {"-":>6s} {"-":>9s} {"-":>12s} '
                     f'{"static":>16s} {"-":>9s} {0:10d} {0.0:8.2f}')
            continue
        hops = f'{np.mean([r["hops"] for r in rr]):6.1f}' if rr else f'{"-":>6s}'
        dly = (f'{np.mean([r["delay_s"] for r in rr]) * 1e3:9.2f}' if rr
               else f'{"-":>9s}')
        npaths = (f'{np.mean([r["n_paths"] for r in rr]):12.2f}' if rr
                  else f'{"-":>12s}')
        ex, exfb = res.get('n_route_exec', 0), res.get('n_route_exec_fallbacks', 0)
        surv = (f'{100 * (ex - exfb) / ex:15.1f}%' if ex else f'{"n/a":>16s}')
        surv_h = (f'{100 * (look - fb) / look:8.1f}%' if look
                  else f'{"n/a":>9s}')
        L.append(f'  {name:16s} {hops} {dly} {npaths} {surv} {surv_h} '
                 f'{len(rlog):10d} {sum(x["wall_s"] for x in rlog):8.2f}')
    L.append('  (frozen ok @exec = a route frozen for a macro-epoch was still')
    L.append('   feasible at the slot it was USED; @horizon = across the whole')
    L.append('   planning lookahead. The price of the hierarchical coupling on')
    L.append('   a MOVING constellation -- GSL contacts last only ~230-265 s.)')
    L.append('')

    L.append('── Delay / depth mix ────────────────────────────────────────')
    for name, res in results.items():
        tk = res['tasks']
        delays = [k.completion_slot * C.SLOT_S - k.arrival_slot * C.SLOT_S
                 for k in tk if k.completion_slot is not None]
        mean_d = np.mean(delays) if delays else float('nan')
        p95_d = np.percentile(delays, 95) if delays else float('nan')
        mix = ' '.join(f'{q}:{sum(1 for k in tk if k.depth == q)}' for q in C.DEPTHS)
        L.append(f'  {name:16s} mean delay {mean_d:7.0f}s  p95 {p95_d:7.0f}s  '
                 f'depth mix {mix}')
    L.append('')

    L.append('── MPC solve-time instrumentation ───────────────────────────')
    for name, res in results.items():
        log = res.get('solve_log') or []
        if not log:
            continue
        ms = [r['wall_s'] * 1e3 for r in log]
        L.append(f'  {name:16s} {len(log):4d} solves   '
                 f'mean {np.mean(ms):7.1f} ms   p95 {np.percentile(ms, 95):7.1f} ms   '
                 f'max {np.max(ms):7.1f} ms   total {sum(ms) / 1e3:6.1f} s')
    L.append('')

    L.append('Outputs in oec_scenario/: contact_windows.csv, topology_state.csv,')
    L.append('tasks.csv, timeline_*.csv, task_outcomes_*.csv, solve_log_*.csv,')
    L.append('plots/*.png, viewer.html (interactive simulator)')
    L.append('=' * 74)
    return '\n'.join(L)


GOLDEN_PATH = os.path.join(C.OUT_DIR, 'golden', 'legacy_summary.json')


def golden_snapshot(results, oracle_txt=None):
    """Machine-readable fingerprint of a run, for the legacy regression gate."""
    import json  # noqa: F401  (kept local; json is only needed for the gate)
    snap = {'utility_mode': C.UTIL_MODE, 'scenario': C.SCENARIO,
            'schedulers': {}}
    for name, r in sorted(results.items()):
        tasks = r['tasks']
        snap['schedulers'][name] = {
            'utility': round(r['hist']['utility'][-1], 6),
            'images': round(r['hist']['delivered_images'][-1], 3),
            'depth_mix': {str(q): sum(1 for k in tasks if k.depth == q)
                          for q in C.DEPTHS},
            'dropped': sum(1 for k in tasks if k.dropped),
        }
    if oracle_txt:
        for line in oracle_txt.splitlines():
            for key, tag in (('analytic ceiling', 'analytic_ceiling'),
                             ('LP bound', 'lp_bound')):
                if line.strip().startswith(key):
                    try:
                        snap[tag] = float(line.split(':')[1].split()[0])
                    except (IndexError, ValueError):
                        pass
    return snap


def check_golden(snap, path=None, tol=1e-4):
    """Assert `snap` matches the committed golden file. Returns a report
    string and a bool; the caller exits non-zero on failure."""
    import json
    # resolved at call time, not bound as a default: --out-suffix rewrites
    # GOLDEN_PATH after this module is imported
    path = path or GOLDEN_PATH
    if not os.path.exists(path):
        return f'!! no golden file at {path} -- write one with --write-golden', False
    with open(path) as fh:
        gold = json.load(fh)
    bad, skipped = [], []
    if gold.get('utility_mode') != snap.get('utility_mode'):
        bad.append(f"utility_mode {gold.get('utility_mode')} != {snap.get('utility_mode')}")
    for name, g in gold.get('schedulers', {}).items():
        got = snap['schedulers'].get(name)
        if got is None:
            # not run (e.g. mpc-hier without --hier). Not drift -- say so
            # rather than failing, so a bare --check-golden stays usable.
            skipped.append(name)
            continue
        for field in ('utility', 'images'):
            if abs(got[field] - g[field]) > max(tol, tol * abs(g[field])):
                bad.append(f'{name}.{field}: {got[field]} != {g[field]} (golden)')
        if got['depth_mix'] != g['depth_mix']:
            bad.append(f"{name}.depth_mix: {got['depth_mix']} != {g['depth_mix']}")
        if got['dropped'] != g['dropped']:
            bad.append(f"{name}.dropped: {got['dropped']} != {g['dropped']}")
    for tag in ('analytic_ceiling', 'lp_bound'):
        if tag in gold and tag in snap and abs(gold[tag] - snap[tag]) > 1e-2:
            bad.append(f'{tag}: {snap[tag]} != {gold[tag]} (golden)')
    if bad:
        return ('!! GOLDEN CHECK FAILED (' + str(len(bad)) + ' mismatches)\n  '
                + '\n  '.join(bad)), False
    n = len(gold.get('schedulers', {})) - len(skipped)
    msg = f'golden check passed: {n} schedulers match {os.path.relpath(path)}'
    if skipped:
        msg += f"  (not run, so not checked: {', '.join(sorted(skipped))})"
    return msg, True


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', default=C.SCENARIO,
                   choices=list(C._SCENARIOS))
    p.add_argument('--congestion', action='store_true',
                   help='also run mpc-congestion (predictive routing, '
                        'Dr. Liu\'s directive) alongside the flat mpc')
    p.add_argument('--hier', action='store_true',
                   help='also run mpc-hier (hierarchical MPC)')
    p.add_argument('--twolevel', action='store_true',
                   help='also run mpc-2level (routing MPC and depth MPC as '
                        'iterated peers)')
    p.add_argument('--hier-route', action='store_true',
                   help='also run mpc-hier-route (slow routing MPC freezing '
                        'a path set for a fast depth MPC)')
    p.add_argument('--couplings', action='store_true',
                   help='shorthand for --twolevel --hier-route --hier')
    p.add_argument('--oracle', action='store_true',
                   help='compute the HiGHS offline upper bound and append '
                        'the optimality-gap table to summary.txt')
    p.add_argument('--utility', default=C.UTIL_MODE, choices=('legacy', 'unified'),
                   help="'legacy' reproduces the committed single-factor "
                        "score exactly; 'unified' turns on the four-factor "
                        'score (quality/timeliness/coverage/cost + fairness)')
    p.add_argument('--ppo-legacy', default=None)
    p.add_argument('--ppo-unified', default=None)
    p.add_argument('--check-golden', action='store_true',
                   help='assert the run reproduces oec_scenario/golden/'
                        'legacy_summary.json to 1e-4 and exit non-zero on '
                        'drift (regression gate for the committed numbers)')
    p.add_argument('--write-golden', action='store_true',
                   help='(re)write the golden file from this run')
    p.add_argument('--out-suffix', default='',
                   help='write outputs to oec_scenario<suffix>/ instead, so a '
                        'second scenario (e.g. the coupling comparison) can be '
                        'committed alongside the canonical run')
    args = p.parse_args(argv)
    C.apply_scenario(args.scenario)
    C.apply_utility_mode(args.utility)
    if args.out_suffix:
        global GOLDEN_PATH
        C.OUT_DIR = C.OUT_DIR + args.out_suffix
        os.makedirs(C.OUT_DIR, exist_ok=True)
        GOLDEN_PATH = os.path.join(C.OUT_DIR, 'golden', 'legacy_summary.json')

    print(f'Building topology... (scenario={args.scenario})')
    t0 = time.time()
    topo = T.build_topology()
    print(f'  done in {time.time() - t0:.1f} s '
          f'({C.N_SATS} sats, {len(topo.isl_pairs)} ISLs, {C.N_SLOTS} slots)')

    extra_makers = []
    if args.ppo_legacy or args.ppo_unified:
        from .rl_train import RLScheduler

        def make_rl(path, name):
            def _make(tp, tk):
                sched = RLScheduler(tp, tk, model_path=path)
                sched.name = name
                return sched
            return _make

        if args.ppo_legacy:
            extra_makers.append(make_rl(args.ppo_legacy, 'ppo-legacy'))
        if args.ppo_unified:
            extra_makers.append(make_rl(args.ppo_unified, 'ppo-unified'))
    if args.congestion:
        extra_makers.append(lambda tp, tk: MPCScheduler(tp, tk, route_mode='predictive'))
    if args.couplings:
        args.hier = args.twolevel = args.hier_route = True
    if args.hier:
        from .hier import HierarchicalMPCScheduler
        extra_makers.append(lambda tp, tk: HierarchicalMPCScheduler(tp, tk))
    if args.twolevel:
        from .twolevel import TwoLevelMPCScheduler
        extra_makers.append(lambda tp, tk: TwoLevelMPCScheduler(tp, tk))
    if args.hier_route:
        from .hier import HierRouteMPCScheduler
        extra_makers.append(lambda tp, tk: HierRouteMPCScheduler(tp, tk))

    print('Running schedulers...')
    results = run_schedulers(topo, extra_makers=extra_makers)

    oracle_txt = None
    if args.oracle:
        from . import oracle as OR
        any_tasks = next(iter(results.values()))['tasks']
        print('Solving oracle upper bound...')
        oracle_txt = OR.report(topo, any_tasks, results)
        with open(os.path.join(C.OUT_DIR, 'upper_bound.txt'), 'w') as f:
            f.write(oracle_txt)
        print(oracle_txt)

    print('Writing outputs...')
    write_outputs(topo, results)

    summary = build_summary(topo, results)
    if oracle_txt:
        summary += '\n\n' + oracle_txt
    with open(os.path.join(C.OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)
    print()
    print(summary)

    if args.write_golden or args.check_golden:
        import json
        snap = golden_snapshot(results, oracle_txt)
        if args.write_golden:
            os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
            with open(GOLDEN_PATH, 'w') as f:
                json.dump(snap, f, indent=2, sort_keys=True)
            print(f'wrote golden -> {os.path.relpath(GOLDEN_PATH)}')
        if args.check_golden:
            msg, ok = check_golden(snap)
            print(msg)
            if not ok:
                raise SystemExit(1)

    try:
        from . import plots
        plots.make_all(topo, results)
    except ImportError:
        print('plots.py not available, skipping')
    try:
        from . import viewer
        viewer.write_viewer(os.path.join(C.OUT_DIR, 'viewer.html'))
    except ImportError:
        print('viewer.py not available, skipping')


if __name__ == '__main__':
    main()
