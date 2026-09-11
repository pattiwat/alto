"""Soft Actor-Critic with Lagrangian constraints, for the Floor 8 grey-box.

    agent = SACAgent(obs_dim, act_dim, SACConfig())
    agent.observe(obs, act, reward, costs, next_obs, terminated, truncated)
    agent.update()
    a = agent.act(obs, deterministic=True)

Written against torch directly rather than stable-baselines3. Every `block-*-derivation/reproduce_*.py`
in this repository is self-contained because there is no `src/` package to import, and SB3 has no
CMDP support - the dual-ascent loop below has to be hand-written under either choice - so the
framework would buy a black box and little else. The hyperparameters are stated in `SACConfig` so
the result stays comparable to published work, which is the one thing a framework would have given.

WHY SAC, AND NOT SOMETHING ELSE
-------------------------------
Each reason is tied to a measured fact about this record, not to general practice
(rl-environment-design 2.1):

  1. The action is continuous, bounded and 4-dimensional - SAC/TD3 territory.
  2. The maximum-entropy objective is the correct prior given F2: the logged policy sat on ONE
     setpoint value 85% of the time, and the replay buffer is seeded from it. A deterministic method
     (DDPG, TD3) collapses onto that band early and never discovers AHU-1's headroom, which is the
     one real asymmetry in this building. The entropy bonus keeps the policy spread long enough to
     find it.
  3. Automatic temperature tuning removes the one hyperparameter there is no budget to tune: with
     ~62 real episodes there is no held-out set large enough to sweep alpha by hand.
  4. Off-policy replay lets the logged transitions seed the buffer, so the agent starts from the
     plant's own behaviour rather than from noise - which matters because the surrogate is least
     trustworthy exactly where the plant has never been.

CONSTRAINTS ARE CONSTRAINTS, NOT WEIGHTS
----------------------------------------
Comfort and minimum airflow enter as a CMDP with multipliers learned by dual ascent, not as a
hand-weighted penalty. The practical difference is not subtle: a weight trades comfort for energy at
whatever exchange rate it implies, and the tuner discovers that rate only after the agent has
exploited it. A constraint says the trade is not available. F5 is explicit that on this floor the
prize is largely ventilation reduction wearing an efficiency costume, so a scalarised comfort term
is an invitation the agent will accept.

The buffer stores reward and costs SEPARATELY and the Lagrangian is applied at sample time. If the
augmented reward were stored, every multiplier update would leave the buffer stale.

TWO IMPLEMENTATION DETAILS THAT ARE EASY TO GET WRONG
-----------------------------------------------------
  * BOOTSTRAP ON TRUNCATION. The target is zeroed only on `terminated`, never on `truncated`. In
    this environment `terminated` is always False - a building day has no absorbing state - so the
    value function must bootstrap through the end of every episode. Marking the horizon terminal
    trains fine and converges to the wrong thing: a critic that believes the world ends at midnight
    and a policy that is myopic near close of day.
  * LAYERNORM ON THE CRITICS. rl-environment-design 3 calls it "the single highest-value trick at
    this scale": it stabilises value estimation and acts as implicit pessimism against
    out-of-distribution actions, which is F2's failure mode exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


@dataclass
class SACConfig:
    hidden: tuple[int, ...] = (256, 256)   # 3: matched to ~1e3 real transitions; capacity here
    n_critics: int = 5                     #    buys memorisation, not generalisation
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    lr_alpha: float = 3e-4
    # Dual ascent now runs ONCE PER EPISODE, not once per gradient step, so this rate is per
    # episode and has to be correspondingly larger - 1e-3 per episode would need thousands of
    # episodes to move the multiplier at all. See `_update_duals`.
    lr_dual: float = 0.05
    batch_size: int = 256
    buffer_size: int = 1_000_000
    warmup_steps: int = 2_000
    updates_per_step: int = 1
    target_entropy: float | None = None    # defaults to -act_dim
    # Constraint budgets, in the units the env emits. `cost_comfort` is summed squared zone-K over
    # a scored step; `cost_vent` counts AHU-steps where the fan cannot supply sum(V_min).
    comfort_budget: float = 1.0
    vent_budget: float = 0.0
    lam_init: float = 1.0
    # On the NORMALISED violation scale (see `_norm_c`), lam is a pure exchange rate between
    # "one budget's worth of discomfort" and "one episode's worth of energy". 1e3 was three
    # orders of magnitude above anything meaningful: at lam_c = 1000 the constraint term buried
    # the energy reward and the critic was fitting the multiplier rather than the plant.
    lam_max: float = 20.0
    seed: int = 0
    device: str = "cpu"


def _mlp(sizes, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed Gaussian. The squash enforces the action bounds STRUCTURALLY.

    That matters here: 9's clip is the boundary of where the environment's own equations were
    identified, so an action that could exceed it and be clipped afterwards would leave the policy
    gradient pointing outside the support.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden):
        super().__init__()
        self.trunk = _mlp((obs_dim, *hidden, 2 * act_dim))
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False, with_logprob=True):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        if not with_logprob:
            return a, None
        # log-prob with the tanh Jacobian, in the numerically stable form
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return a, logp


class CriticEnsemble(nn.Module):
    """N independent Q heads, aggregated pessimistically.

    The ensemble buys sample efficiency at a higher update-to-data ratio AND supplies the epistemic
    uncertainty F4 says this environment genuinely has - beta's interval is wider than the effect it
    measures.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden, n: int):
        super().__init__()
        self.qs = nn.ModuleList([_mlp((obs_dim + act_dim, *hidden, 1), layer_norm=True)
                                 for _ in range(n)])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return torch.cat([q(x) for q in self.qs], dim=-1)      # (B, n)


class ReplayBuffer:
    """Flat arrays. Reward and the two costs are stored SEPARATELY - see the module docstring."""

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.c_comfort = np.zeros(size, dtype=np.float32)
        self.c_vent = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)           # `terminated` ONLY, never truncated
        self.ptr, self.full, self.size = 0, False, size

    def add(self, o, a, r, cc, cv, o2, terminated):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.c_comfort[i], self.c_vent[i] = cc, cv
        self.next_obs[i], self.done[i] = o2, float(terminated)
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self) -> int:
        return self.size if self.full else self.ptr

    def sample(self, batch: int, rng: np.random.Generator, device):
        idx = rng.integers(0, len(self), size=batch)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.obs), t(self.act), t(self.rew), t(self.c_comfort), t(self.c_vent), \
            t(self.next_obs), t(self.done)


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig | None = None):
        self.cfg = cfg or SACConfig()
        c = self.cfg
        torch.manual_seed(c.seed)
        self.rng = np.random.default_rng(c.seed)
        self.dev = torch.device(c.device)
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.actor = Actor(obs_dim, act_dim, c.hidden).to(self.dev)
        self.critic = CriticEnsemble(obs_dim, act_dim, c.hidden, c.n_critics).to(self.dev)
        self.critic_targ = CriticEnsemble(obs_dim, act_dim, c.hidden, c.n_critics).to(self.dev)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=c.lr)

        # Auto-tuned entropy temperature. Target -dim(A) is the standard choice and there is no
        # budget to sweep it (2.1, reason 3).
        self.target_entropy = c.target_entropy if c.target_entropy is not None else -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.dev)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=c.lr_alpha)

        # Lagrange multipliers, kept in log-space so they cannot go negative.
        self.log_lam_comfort = torch.tensor(np.log(c.lam_init), requires_grad=True, device=self.dev)
        self.log_lam_vent = torch.tensor(np.log(c.lam_init), requires_grad=True, device=self.dev)
        self.opt_dual = torch.optim.Adam([self.log_lam_comfort, self.log_lam_vent], lr=c.lr_dual)

        # CONSTRAINT NORMALISERS. The multiplier must multiply a DIMENSIONLESS violation, not a
        # raw one. Comfort cost is summed squared zone-K over 53 zones and runs to hundreds per
        # step on the two-sided metric, while a step's energy reward is ~1.5 - so an un-normalised
        # `r - lam*cc` is dominated by the constraint at any lam above ~0.01, and the dual
        # gradient `lam*(J_c - budget)` moves log_lam by thousands in a single step. Dividing both
        # the sample-time penalty and the dual gradient by the budget puts one budget's worth of
        # violation at 1.0, which is what makes lam interpretable and lam_max meaningful.
        self._norm_c = max(float(c.comfort_budget), 1e-6)
        self._norm_v = max(float(c.vent_budget), 1.0)

        self.buf = ReplayBuffer(obs_dim, act_dim, c.buffer_size)
        self.total_steps = 0
        self.updates = 0
        self._ep_costs: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ acting
    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @property
    def lam(self) -> tuple[float, float]:
        return (float(self.log_lam_comfort.exp().item()),
                float(self.log_lam_vent.exp().item()))

    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.total_steps < self.cfg.warmup_steps:
            # Uniform warm-up. The buffer is ALSO seeded from the logged transitions, so this is
            # exploration around a real operating point rather than from nothing.
            return self.rng.uniform(-1.0, 1.0, size=self.act_dim)
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.dev).unsqueeze(0)
            a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def observe(self, o, a, r, c_comfort, c_vent, o2, terminated: bool, truncated: bool) -> None:
        # `truncated` is deliberately unused in the stored flag: bootstrapping must continue through
        # the horizon. It is in the signature so a caller cannot silently pass it as `terminated`.
        assert not terminated, ("terminated must be False in this environment - a building day has "
                                "no absorbing state. Zeroing the bootstrap here would teach the "
                                "critic that the world ends at the horizon.")
        self.buf.add(np.asarray(o, np.float32), np.asarray(a, np.float32), r,
                     c_comfort, c_vent, np.asarray(o2, np.float32), terminated)
        self.total_steps += 1

    def note_episode_costs(self, comfort: float, vent: float) -> None:
        """Episode constraint returns feed the dual update - the constraint is on the RETURN.

        The dual step is taken HERE, once per episode. It used to be taken from `update()`, i.e.
        once per environment step - 150,000 times over a run, every one of them on a statistic
        that only changes at the episode boundary. The multiplier was re-solving the same
        constraint hundreds of times per episode and slamming into its cap between them.
        """
        self._ep_costs.append((comfort, vent))
        self._update_duals()

    # ------------------------------------------------------------------ learning
    def update(self) -> dict:
        c = self.cfg
        if len(self.buf) < max(c.batch_size, c.warmup_steps):
            return {}
        logs = {}
        for _ in range(c.updates_per_step):
            logs = self._update_once()
        # NO dual step here - it belongs to the episode boundary, in `note_episode_costs`.
        return logs

    def _update_once(self) -> dict:
        c = self.cfg
        o, a, r, cc, cv, o2, d = self.buf.sample(c.batch_size, self.rng, self.dev)
        lam_c, lam_v = self.log_lam_comfort.exp().detach(), self.log_lam_vent.exp().detach()
        # The Lagrangian is applied HERE, at sample time, so a multiplier update never leaves the
        # stored reward stale.
        r_eff = r - lam_c * (cc / self._norm_c) - lam_v * (cv / self._norm_v)

        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q2 = self.critic_targ(o2, a2).min(dim=-1).values         # pessimistic aggregate
            backup = r_eff + c.gamma * (1.0 - d) * (q2 - self.alpha * logp2)

        q = self.critic(o, a)
        loss_q = F.mse_loss(q, backup.unsqueeze(-1).expand_as(q))
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_critic.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        pi, logp = self.actor(o)
        q_pi = self.critic(o, pi).min(dim=-1).values
        loss_pi = (self.alpha * logp - q_pi).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.opt_actor.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1.0 - c.tau).add_(c.tau * p)

        self.updates += 1
        return {"loss_q": float(loss_q.item()), "loss_pi": float(loss_pi.item()),
                "alpha": self.alpha, "q_mean": float(q.mean().item())}

    def _update_duals(self) -> None:
        """lam <- [lam + eta*(J_c - budget)/norm]_+ , on the running episode constraint return.

        Called once per EPISODE from `note_episode_costs`. Slower than the primal by design: a
        multiplier that chases the critic makes the augmented reward non-stationary and the value
        estimate follows it around.

        The violation is divided by the budget so it is dimensionless - "fractions of a budget
        overspent" rather than "squared zone-kelvin". Without that the gradient scales with the
        raw cost, which on the two-sided metric is O(1e3), and a single step moves log_lam by
        more than the entire useful range of the multiplier.
        """
        if len(self._ep_costs) < 5:
            return
        comfort = float(np.mean([x[0] for x in self._ep_costs[-20:]]))
        vent = float(np.mean([x[1] for x in self._ep_costs[-20:]]))
        c = self.cfg
        viol_c = (comfort - c.comfort_budget) / self._norm_c
        viol_v = (vent - c.vent_budget) / self._norm_v
        loss = -(self.log_lam_comfort.exp() * viol_c + self.log_lam_vent.exp() * viol_v)
        self.opt_dual.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_dual.step()
        with torch.no_grad():
            cap = float(np.log(c.lam_max))
            self.log_lam_comfort.clamp_(-20.0, cap)
            self.log_lam_vent.clamp_(-20.0, cap)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "log_alpha": self.log_alpha.detach(),
                    "log_lam": [self.log_lam_comfort.detach(), self.log_lam_vent.detach()],
                    "cfg": self.cfg.__dict__, "steps": self.total_steps}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_targ.load_state_dict(ck["critic"])
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"])
            self.log_lam_comfort.copy_(ck["log_lam"][0])
            self.log_lam_vent.copy_(ck["log_lam"][1])


def seed_buffer_from_record(agent: SACAgent, env, day_indices, policy) -> int:
    """SACfD-style: fill the buffer with the plant's own behaviour before learning starts.

    2.1 reason 4. The surrogate is least trustworthy exactly where the plant has never been, so
    starting the critic from the region the record actually covers is worth more here than the
    extra transitions themselves.
    """
    n = 0
    for i in day_indices:
        o, _ = env.reset(seed=agent.cfg.seed + i, options={"day": i})
        for _ in range(env.horizon):
            a = policy(o, env)
            o2, r, term, trunc, info = env.step(a)
            agent.buf.add(np.asarray(o, np.float32), np.asarray(a, np.float32), r,
                          info["cost_comfort"], info["cost_vent"],
                          np.asarray(o2, np.float32), False)
            o = o2
            n += 1
    return n
