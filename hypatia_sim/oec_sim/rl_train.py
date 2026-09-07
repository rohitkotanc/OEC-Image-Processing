"""
Train / evaluate the PPO baseline (Dr. Liu: MPC vs. RL comparison).

This needs torch + stable-baselines3 + sb3-contrib, which are NOT installed
on the Mac dev machine (verified 2026-08-20: only numpy/scipy/matplotlib
are present, and this repo's Python is 3.14 -- sb3 wheels lag new Python
releases). Run this on the NCSU server (eb3-2402-grd04, 2x A6000) inside
tmux, in a SEPARATE venv from the main sim:

    python3.11 -m venv hypatia_sim/.venv-rl
    source hypatia_sim/.venv-rl/bin/activate
    pip install -r hypatia_sim/requirements.txt -r hypatia_sim/requirements-rl.txt
    tmux new -s oec-ppo
    cd hypatia_sim && python3 -m oec_sim.rl_train train --timesteps 5000000
    # detach: ctrl-b d ; reattach: tmux attach -t oec-ppo

Generalization protocol (see FORMULATION.md): train on GS_RATE_BPS in
{1.5, 2.0, 2.5} Mbps x seeds 0-99; evaluate in-distribution (2.0 Mbps,
seeds 100-119), OOD load (0.75/1.0/3.0/4.0 Mbps), and OOD topology
(starlink-550) without retraining -- the payoff of the per-task,
topology-agnostic observation in rl_env.py.
"""

import argparse
import os

from . import config as C
from . import utility
from . import topology as T
from .schedulers import _Base, _record_delivery


def _require_sb3():
    try:
        import gymnasium  # noqa: F401
        from sb3_contrib import MaskablePPO  # noqa: F401
        from sb3_contrib.common.wrappers import ActionMasker  # noqa: F401
        from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'RL training needs gymnasium + sb3-contrib + stable-baselines3 '
            '+ torch (pip install -r requirements-rl.txt), preferably on '
            'the server in a Python 3.11/3.12 venv -- see this module\'s '
            f'docstring. Missing: {e}')


def train(timesteps=5_000_000, train_rates_mbps=(1.5, 2.0, 2.5),
          n_train_seeds=100, out_path='ppo_oec.zip', utility_mode='legacy'):
    _require_sb3()
    import numpy as np
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .rl_env import make_gym_env, N_ACTIONS
    C.apply_utility_mode(utility_mode)
    print(f'Training PPO with utility={C.UTIL_MODE}')
    topo = T.build_topology()

    def _mask_fn(env):
        return env.action_masks()

    def _make(rate_mbps, seed):
        def _thunk():
            env = make_gym_env(
                topo,
                seed=seed,
                gs_rate_bps=rate_mbps * 1e6
            )
            return ActionMasker(env, _mask_fn)
        return _thunk

    envs = [_make(rate, seed) for rate in train_rates_mbps
           for seed in range(n_train_seeds)]
    vec_env = DummyVecEnv(envs)

    model = MaskablePPO('MlpPolicy', vec_env, verbose=1,
                        n_steps=2048, batch_size=256, learning_rate=3e-4,
                        ent_coef=0.01, gamma=0.999, gae_lambda=0.95,
                        policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=timesteps)
    model.save(out_path)
    print(f'saved -> {out_path}')
    return model


class RLScheduler(_Base):
    """Wraps a trained MaskablePPO policy as a schedulers._Base subclass so
    it drops into run_all.run_schedulers() and gets identical metrics/CSVs
    to every other scheduler."""

    name = 'ppo'

    def __init__(self, topo, tasks, model_path='ppo_oec.zip'):
        super().__init__(topo, tasks)
        _require_sb3()
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        from .rl_env import OECDepthEnv
        self._env = OECDepthEnv(topo, tasks=tasks)

    def run(self):
        import time as _time
        from .rl_env import N_ACTIONS
        hist = {'t_s': [], 'delivered_images': [], 'utility': [],
               'backlog_bits': [], 'n_active': [], 'n_dropped': []}
        obs, _ = self._env.reset()
        env = self._env
        while env.t < C.N_SLOTS:
            if env._active_queue and env._active_queue[0].depth is None:
                mask = env.action_masks()
                t0 = _time.perf_counter()
                action, _ = self.model.predict(obs, action_masks=mask,
                                                deterministic=True)
                self.solve_log.append(dict(t_s=env.t * C.SLOT_S,
                                          wall_s=_time.perf_counter() - t0))
                obs, r, terminated, truncated, info = env.step(int(action))
            else:
                obs, r, terminated, truncated, info = env.step(4)  # defer/advance
            if not hist['t_s'] or hist['t_s'][-1] != env.t * C.SLOT_S:
                active = [k for k in self.tasks
                         if k.arrival_slot <= env.t and k.delivered < k.n_images - 1e-6
                         and not k.dropped]
                backlog = sum((k.n_images - k.delivered)
                             * C.PAYLOAD_B[k.depth or max(C.DEPTHS)] * 8
                             for k in self.tasks
                             if k.arrival_slot <= env.t and not k.dropped)
                hist['t_s'].append(env.t * C.SLOT_S)
                hist['delivered_images'].append(sum(k.delivered for k in self.tasks))
                # same accounting as _Base.run(): utility.run_utility carries
                # the fairness bonus, so ppo stays comparable to every other
                # scheduler under --utility unified
                arrived = [k for k in self.tasks if k.arrival_slot <= env.t]
                hist['utility'].append(utility.run_utility(arrived)[0])
                hist['backlog_bits'].append(backlog)
                hist['n_active'].append(len(active))
                hist['n_dropped'].append(sum(1 for k in self.tasks if k.dropped))
            if terminated:
                break
        return hist


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    pt = sub.add_parser('train')
    pt.add_argument('--timesteps', type=int, default=5_000_000)
    pt.add_argument('--out', default='ppo_oec.zip')
    pt.add_argument('--utility', default='legacy',
                    choices=('legacy', 'unified'))
    args = p.parse_args()
    if args.cmd == 'train':
        train(timesteps=args.timesteps, out_path=args.out,
              utility_mode=args.utility)
