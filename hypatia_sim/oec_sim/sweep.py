"""
Systematic congestion sweep across GS_RATE_BPS regimes x task seeds x every
scheduler -- closes the "MPC's advantage shown only by a one-off manual run
that isn't saved in the repo" gap (meeting_script_oec.txt section 6). The
crossover this produces (best FIXED depth flips with load; MPC/MPC-hier
track the upper envelope in every regime) is the headline figure.

Run from hypatia_sim/:
    python3 -m oec_sim.sweep                  # default grid, ~10-20 min
    python3 -m oec_sim.sweep --quick           # 2 rates x 2 seeds, smoke test
"""

import csv
import os
import time

import numpy as np

from . import config as C
from . import topology as T
from . import utility
from . import tasks as TK
from .schedulers import GreedyScheduler, MPCScheduler

RATES_MBPS = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
SEEDS = [42, 43, 44, 45, 46]
PPO_EVAL_SEEDS = list(range(100, 120))
PPO_EVAL_RATES_MBPS = [0.75, 1.0, 2.0, 3.0, 4.0]

# Routing only has leverage where the FABRIC is scarce. Under gbs-limited the
# per-GBS aggregate budget is >=12x tighter than any ISL a route could cross,
# so ISL contention is mathematically unreachable and every routing variant
# reproduces the static one bit-for-bit -- we verified exactly that. The
# coupling grid therefore sweeps the ISL rate under fabric-limited instead.
ISL_RATES_MBPS = [0.5, 0.75, 1.0, 1.5, 2.0]
COUPLING_SEEDS = [42, 43, 44]

OUT_DIR = os.path.join(C.OUT_DIR, 'sweep')


def _row(rate_mbps, seed, sched, tasks, hist, tot_img, wall_s, obj=None,
         rate_key='gs_rate_mbps'):
    img = sum(k.delivered for k in tasks)
    n_viol = sum(1 for k in tasks if k.dropped or k.completion_slot is None
                or (k.completion_slot + 1) * C.SLOT_S > k.deadline_s)
    total, terms, jain, umin = utility.run_utility(tasks)
    solve_s = sum(x['wall_s'] for x in getattr(obj, 'solve_log', []) or [])
    return {
        rate_key: rate_mbps, 'seed': seed, 'scheduler': sched,
        'images_delivered': round(img, 1),
        'delivery_frac': round(img / tot_img, 4),
        'utility': round(hist['utility'][-1], 4),
        'utility_quality': round(terms['quality'], 4),
        'utility_tardy': round(terms['tardiness'], 4),
        'utility_cost': round(terms['cost'], 4),
        'u_min': round(umin, 4), 'jain': round(jain, 4),
        'n_dropped': sum(1 for k in tasks if k.dropped),
        'violation_frac': round(n_viol / len(tasks), 4),
        'n_route_fallbacks': getattr(obj, 'n_route_fallbacks', 0),
        'solve_wall_s': round(solve_s, 2),
        'wall_s': round(wall_s, 2),
    }


def run(rates_mbps=RATES_MBPS, seeds=SEEDS, include_hier=False, quick=False):
    if quick:
        rates_mbps, seeds = rates_mbps[:2], seeds[:2]
    os.makedirs(OUT_DIR, exist_ok=True)
    topo = T.build_topology()          # independent of rate/seed -- build once
    rows = []

    hier_maker = None
    if include_hier:
        from .hier import HierarchicalMPCScheduler
        hier_maker = lambda tp, tk: HierarchicalMPCScheduler(tp, tk)

    for rate in rates_mbps:
        with C.config_override(GS_RATE_BPS=rate * 1e6):
            for seed in seeds:
                with C.config_override(RNG_SEED=seed):
                    makers = ([('mpc', lambda tp, tk: MPCScheduler(tp, tk)),
                              ('greedy-adaptive',
                               lambda tp, tk: GreedyScheduler(tp, tk, depth=None))]
                              + [(f'greedy-fixed-{q}',
                                 lambda tp, tk, q=q: GreedyScheduler(tp, tk, depth=q))
                                for q in C.DEPTHS])
                    if hier_maker:
                        makers.append(('mpc-hier', hier_maker))
                    for name, make in makers:
                        tk = TK.generate_tasks(topo)
                        tot_img = sum(k.n_images for k in tk)
                        sched = make(topo, tk)
                        t0 = time.time()
                        hist = sched.run()
                        wall_s = time.time() - t0
                        rows.append(_row(rate, seed, name, tk, hist,
                                         tot_img, wall_s, obj=sched))
                        print(f'  rate={rate:4.2f}Mbps seed={seed:3d} '
                             f'{name:16s} util={hist["utility"][-1]:7.2f} '
                             f'({wall_s:5.1f}s)', flush=True)

    # --quick is a 2x2 smoke test, not a result. Writing it to results.csv
    # silently replaces the committed 180-row grid with 24 rows, which is
    # exactly the kind of loss you only notice much later.
    path = os.path.join(OUT_DIR,
                        'results_quick.csv' if quick else 'results.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {os.path.relpath(path)}')

    if not quick:
        try:
            _plot(rows)
        except ImportError:
            print('matplotlib not available, skipping sweep plot')
    return rows


def _plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plots import SCHED_COLOR, GRID

    scheds = sorted({r['scheduler'] for r in rows})
    rates = sorted({r['gs_rate_mbps'] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in scheds:
        util_by_rate = []
        for rate in rates:
            vals = [r['utility'] for r in rows
                    if r['scheduler'] == s and r['gs_rate_mbps'] == rate]
            util_by_rate.append((np.mean(vals), np.std(vals)))
        means = np.array([u for u, _ in util_by_rate])
        stds = np.array([sd for _, sd in util_by_rate])
        col = SCHED_COLOR.get(s, '#6f6e66')
        lw = 2.4 if s in ('mpc', 'mpc-hier') else 1.4
        ax.plot(rates, means, color=col, lw=lw, marker='o', ms=4, label=s)
        ax.fill_between(rates, means - stds, means + stds, color=col, alpha=0.15)
    ax.set_xlabel('GBS aggregate rate (Mbps per station)')
    ax.set_ylabel('utility (mean over seeds, shaded = ±1 std)')
    ax.set_title('MPC vs. best fixed depth across load regimes -- the '
                 'crossover is the headline result', fontsize=10)
    ax.grid(**GRID)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '..', 'plots', 'sweep_utility_vs_load.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {os.path.relpath(out)}')

def run_ppo_eval(legacy_path, unified_path, quick=False):
    from .rl_train import RLScheduler

    rates = PPO_EVAL_RATES_MBPS
    seeds = PPO_EVAL_SEEDS
    if quick:
        rates = [2.0]
        seeds = seeds[:2]

    os.makedirs(OUT_DIR, exist_ok=True)
    topo = T.build_topology()
    rows = []

    for rate in rates:
        with C.config_override(GS_RATE_BPS=rate * 1e6):
            for seed in seeds:
                with C.config_override(RNG_SEED=seed):
                    makers = [
                        ('ppo-legacy', legacy_path),
                        ('ppo-unified', unified_path),
                    ]
                    for name, path in makers:
                        tk = TK.generate_tasks(topo)
                        tot_img = sum(k.n_images for k in tk)
                        sched = RLScheduler(topo, tk, model_path=path)
                        sched.name = name
                        t0 = time.time()
                        hist = sched.run()
                        wall_s = time.time() - t0
                        rows.append(_row(rate, seed, name, tk, hist,
                                         tot_img, wall_s, obj=sched))
                        print(f'  rate={rate:4.2f}Mbps seed={seed:3d} '
                              f'{name:16s} util={hist["utility"][-1]:7.2f} '
                              f'({wall_s:5.1f}s)', flush=True)

    path = os.path.join(OUT_DIR, 'results_ppo_eval.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {os.path.relpath(path)}')
    return rows

def run_couplings(isl_rates_mbps=ISL_RATES_MBPS, seeds=COUPLING_SEEDS,
                  quick=False):
    """Coupling grid, run under `fabric-limited` where routing can matter.

    Deliberately smaller than the utility grid: mpc-2level re-solves a routing
    LP on every re-plan, so a 6x5 grid of it would run for many hours.
    """
    from .hier import HierarchicalMPCScheduler, HierRouteMPCScheduler
    from .twolevel import TwoLevelMPCScheduler
    if quick:
        isl_rates_mbps, seeds = isl_rates_mbps[:2], seeds[:1]
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    out_name = 'results_routing_quick.csv' if quick else 'results_routing.csv'
    with C.config_override(scenario='fabric-limited'):
        topo = T.build_topology()
        for rate in isl_rates_mbps:
            with C.config_override(ISL_RATE_BPS=rate * 1e6):
                for seed in seeds:
                    with C.config_override(RNG_SEED=seed):
                        makers = [
                            ('mpc', lambda tp, tk: MPCScheduler(tp, tk)),
                            ('mpc-2level',
                             lambda tp, tk: TwoLevelMPCScheduler(tp, tk)),
                            ('mpc-hier',
                             lambda tp, tk: HierarchicalMPCScheduler(tp, tk)),
                            ('mpc-hier-route',
                             lambda tp, tk: HierRouteMPCScheduler(tp, tk)),
                        ]
                        for name, make in makers:
                            tk = TK.generate_tasks(topo)
                            tot_img = sum(k.n_images for k in tk)
                            sched = make(topo, tk)
                            t0 = time.time()
                            hist = sched.run()
                            rows.append(_row(rate, seed, name, tk, hist,
                                             tot_img, time.time() - t0,
                                             obj=sched,
                                             rate_key='isl_rate_mbps'))
                            print(f'  isl={rate:4.2f}Mbps seed={seed:3d} '
                                  f'{name:16s} util={hist["utility"][-1]:7.2f} '
                                  f'({time.time() - t0:5.1f}s)', flush=True)
    path = os.path.join(OUT_DIR, out_name)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {os.path.relpath(path)}')
    return rows


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--hier', action='store_true')
    ap.add_argument('--couplings', action='store_true',
                    help='run the routing-coupling grid (ISL rates under '
                         'fabric-limited) instead of the GBS-rate grid')
    ap.add_argument('--utility', default=C.UTIL_MODE,
                    choices=('legacy', 'unified'))
    ap.add_argument('--ppo-eval', action='store_true')
    ap.add_argument('--ppo-legacy', default='ppo_legacy_corrected.zip')
    ap.add_argument('--ppo-unified', default='ppo_unified.zip')
    args = ap.parse_args()
    C.apply_utility_mode(args.utility)
    if args.ppo_eval:
        run_ppo_eval(args.ppo_legacy, args.ppo_unified, quick=args.quick)
    elif args.couplings:
        run_couplings(quick=args.quick)
    else:
        run(include_hier=args.hier, quick=args.quick)
