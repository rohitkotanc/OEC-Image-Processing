"""
Gymnasium environment for the PPO baseline (Dr. Liu: MPC vs. RL comparison).

Action space is fixed-cardinality by construction: the env steps per
(slot, task) DECISION POINT, not per slot. When a task first becomes
active, the agent is asked for one action; once a depth is committed,
transmission for the rest of the episode runs through the same shared
engine (_edge_keys / _cap_bits / _record_delivery from schedulers.py) that
the greedy baselines and MPC use, so the comparison is apples-to-apples
and not an artifact of a differently-implemented RL execution path.

    action_space = Discrete(6)
      0..3  commit depth C.DEPTHS[i] and transmit at max feasible rate
      4     defer (send nothing this decision; re-asked next slot)
      5     drop (abandon the task -- same lever the hierarchical MPC has)

Requires `gymnasium` (`pip install gymnasium`); guarded import so the rest
of oec_sim works without it (see run_all.py's plots/viewer guard pattern).
"""

import numpy as np

from . import config as C
from . import routing as R
from . import utility
from .schedulers import _edge_keys, _cap_bits, _record_delivery

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                    # pragma: no cover
    gym = None
    spaces = None

OBS_DIM = 16   # 12 task features (incl. 4-way depth one-hot) + 4 context
N_ACTIONS = 6
DEFER, DROP = 4, 5


def _task_obs(k, t, topo, isl_index):
    now_s = t * C.SLOT_S
    rem_frac = (k.n_images - k.delivered) / k.n_images
    encoded_frac = np.clip(k.encoded_by(now_s) / k.n_images, 0.0, 1.0)
    time_to_dl = np.clip((k.deadline_s - now_s) / C.TASK_DEADLINE_MAX_S, -2.0, 2.0)
    phi = k.freshness(now_s)
    keys = _edge_keys(topo, isl_index, t, k)
    has_route = 1.0 if keys is not None else 0.0
    hops = (len(keys) / 30.0) if keys is not None else 1.0
    depth_onehot = [1.0 if k.depth == q else 0.0 for q in C.DEPTHS]
    return np.array([
        rem_frac, np.log10(max(k.n_images, 1)) / 6.0, k.weight / 3.0,
        time_to_dl, phi, encoded_frac, has_route, hops, *depth_onehot,
    ], dtype=np.float32)


class OECDepthEnv:
    """Per-(slot, task) decision-point env. Not a full gymnasium.Env
    subclass at import time (gymnasium may be absent); becomes one via
    _make_gym_env() below when gymnasium is installed."""

    def __init__(self, topo, seed=None, tasks=None):
        self.topo = topo
        self.isl_index = {(min(a, b), max(a, b)): l
                          for l, (a, b) in enumerate(topo.isl_pairs)}
        self.static = R.StaticRouter(topo)
        self._seed = seed
        self._tasks_override = tasks
        self.reset(seed=seed)

    def reset(self, seed=None):
        from . import tasks as TK
        if seed is not None:
            self._seed = seed
        if self._tasks_override is not None:
            self.tasks = self._tasks_override
            for k in self.tasks:
                utility.reset(k)
        else:
            with C.config_override(RNG_SEED=self._seed if self._seed is not None
                                   else C.RNG_SEED):
                self.tasks = TK.generate_tasks(self.topo)
        self.t = 0
        self._active_queue = None       # None = "not yet built for this slot"
        self._prev_potential = 0.0
        self._advance_to_next_decision()
        return self._obs(), {}

    def _potential(self):
        backlog = sum((k.n_images - k.delivered) * C.PAYLOAD_B[k.depth or 16] * 8
                      for k in self.tasks if k.arrival_slot <= self.t and not k.dropped)
        return -1e-3 * backlog / 1e11

    def _advance_to_next_decision(self):
        """Advance slots, executing already-committed tasks with the shared
        engine, until a task needs a depth decision (or the episode ends).

        Uses `self._active_queue is None` (not "falsy") as the "need to
        (re)build this slot's queue" signal. A deferred task pops the queue
        down to `[]`, which is deliberately NOT None: without this
        distinction, an empty-but-not-yet-advanced queue gets immediately
        rebuilt from the *same* slot's active tasks, and a deferred task
        (still undecided) reappears at the same t forever -- a real
        infinite loop whenever the policy consistently defers one task
        (found empirically: a trained policy stuck asking the same task at
        the same slot for 3000+ decisions with t never advancing).
        """
        while self.t < C.N_SLOTS:
            if self._active_queue is None:
                self._active_queue = sorted(
                    (k for k in self.tasks
                     if k.arrival_slot <= self.t and k.delivered < k.n_images - 1e-6
                     and not k.dropped),
                    key=lambda k: k.deadline_s)
                self._residual = {}
            while self._active_queue:
                k = self._active_queue[0]
                if k.depth is None:
                    return                                # ask the agent
                self._execute_one(k)
                self._active_queue.pop(0)
            self._active_queue = None                     # drained -> advance
            self.t += 1
        self._done = True

    def _execute_one(self, k):
        keys = _edge_keys(self.topo, self.isl_index, self.t, k)
        if keys is None:
            return
        for key in keys:
            self._residual.setdefault(key, _cap_bits(key))
        img_bits = C.PAYLOAD_B[k.depth] * 8
        y = min(k.encoded_by((self.t + 1) * C.SLOT_S) - k.delivered,
               min(self._residual[key] for key in keys) / max(img_bits, 1e-9))
        if y <= 1e-9:
            return
        for key in keys:
            self._residual[key] -= y * img_bits
        # Charge the real propagation delay, as the MPC and greedy paths do.
        # Passing no delay here scored RL deliveries as if propagation were
        # instantaneous, which inflated their freshness and made the PPO
        # baseline look better than the schedulers it is compared against.
        _record_delivery(k, self.t, y,
                         route_delay_s=self.static.delay_s(
                             self.t, k.dst_gs, k.src_sat))

    def _obs(self):
        if self.t >= C.N_SLOTS or not self._active_queue:
            return np.zeros(OBS_DIM, dtype=np.float32)
        k = self._active_queue[0]
        task_feat = _task_obs(k, self.t, self.topo, self.isl_index)      # 15
        active = [kk for kk in self.tasks
                 if kk.arrival_slot <= self.t and kk.delivered < kk.n_images - 1e-6
                 and not kk.dropped]
        n_same_gs = sum(1 for kk in active if kk.dst_gs == k.dst_gs)
        ctx = np.array([
            self.t / C.N_SLOTS, len(active) / 16.0, n_same_gs / 8.0,
            np.mean([kk.freshness(self.t * C.SLOT_S) for kk in active])
            if active else 1.0,
        ], dtype=np.float32)
        return np.concatenate([task_feat, ctx])[:OBS_DIM].astype(np.float32)

    def action_masks(self):
        if self.t >= C.N_SLOTS or not self._active_queue:
            return np.zeros(N_ACTIONS, dtype=bool)
        k = self._active_queue[0]
        keys = _edge_keys(self.topo, self.isl_index, self.t, k)
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if keys is not None:
            mask[:len(C.DEPTHS)] = True
        mask[DEFER] = True
        mask[DROP] = True
        return mask

    def step(self, action):
        k = self._active_queue[0]
        u_before = k.delivered_utility
        if action < len(C.DEPTHS):
            k.depth = C.DEPTHS[int(action)]
            self._execute_one(k)
            self._active_queue.pop(0)
        elif action == DEFER:
            self._active_queue.pop(0)
        elif action == DROP:
            k.dropped = True
            k.dropped_slot = self.t
            self._active_queue.pop(0)
        dJ = k.delivered_utility - u_before

        self._done = False
        self._advance_to_next_decision()
        potential = self._potential()
        reward = dJ + (0.999 * potential - self._prev_potential)
        self._prev_potential = potential
        terminated = self.t >= C.N_SLOTS
        return self._obs(), reward, terminated, False, {}


def make_gym_env(topo, seed=None, tasks=None, gs_rate_bps=None):
    """Wrap OECDepthEnv as a real gymnasium.Env (needs `pip install
    gymnasium`); raises ImportError with an actionable message otherwise."""
    if gym is None:
        raise ImportError('gymnasium is not installed -- '
                          'pip install -r requirements-rl.txt')

    class _GymOECDepthEnv(gym.Env):
        metadata = {'render_modes': []}

        def __init__(self):
            super().__init__()
            self._gs_rate_bps = gs_rate_bps
            if self._gs_rate_bps is None:
                self._env = OECDepthEnv(topo, seed=seed, tasks=tasks)
            else:
                with C.config_override(GS_RATE_BPS=self._gs_rate_bps):
                    self._env = OECDepthEnv(topo, seed=seed, tasks=tasks)

            self.observation_space = spaces.Box(
                -10.0,
                10.0,
                shape=(OBS_DIM,),
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(N_ACTIONS)

        def reset(self, *, seed=None, options=None):
            if self._gs_rate_bps is None:
                return self._env.reset(seed=seed)

            with C.config_override(GS_RATE_BPS=self._gs_rate_bps):
                return self._env.reset(seed=seed)

        def step(self, action):
            if self._gs_rate_bps is None:
                return self._env.step(action)

            with C.config_override(GS_RATE_BPS=self._gs_rate_bps):
                return self._env.step(action)

        def action_masks(self):
            if self._gs_rate_bps is None:
                return self._env.action_masks()

            with C.config_override(GS_RATE_BPS=self._gs_rate_bps):
                return self._env.action_masks()

    return _GymOECDepthEnv()

