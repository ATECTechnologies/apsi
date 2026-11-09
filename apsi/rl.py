"""`use rl` — reinforcement learning, learning from preferences (RLHF, DPO, GRPO) and search.

Environments (vectorised, NumPy only)
  cartpole(n=1, seed)       the classic pole-balancing task (dynamics identical to Gymnasium's CartPole-v1)
  gridworld(size, n, seed)  move to the goal on a grid with a step penalty
  bandit(arms, n, seed)     multi-armed Gaussian bandit
  Every environment has reset(seed) -> obs [n, …] and step(actions) -> record(obs, reward, done, truncated,
  final_obs); finished environments reset themselves.
Returns and advantages
  discounted_returns(rewards, dones, gamma, last_value), gae(rewards, values, dones, last_values, gamma, lam)
Policies
  categorical_sample(logits, seed), categorical_log_prob(logits, actions), categorical_entropy(logits),
  gaussian_log_prob(mean, log_std, actions), gaussian_entropy(log_std), sample_token(logits, temperature, top_k,
  top_p, seed)
Algorithms
  ppo(env, model, params, total_steps, …)    clipped PPO with GAE, value loss, entropy bonus, gradient clipping;
                                              model(params, obs) returns [logits, value]
  dqn(env, model, params, total_steps, …)    double DQN with replay buffer, target network and Huber loss
  reinforce(env, model, params, episodes, …) policy gradient with a baseline
  evaluate(env, policy_fn, episodes, seed)
Learning from preferences
  preference_loss(r_chosen, r_rejected)        Bradley–Terry reward-model loss
  dpo_loss(pc, pr, rc, rr, beta)               Direct Preference Optimization (log-probs of chosen / rejected)
  ipo_loss(pc, pr, rc, rr, tau)                Identity Preference Optimization
  kl_penalized_rewards(rewards, logp, ref_logp, beta)   RLHF reward with a per-token KL penalty
  grpo_advantages(rewards, group_size)         group-relative advantages (GRPO)
  ppo_clip_loss(logp, old_logp, adv, clip)     the clipped surrogate for sequence models
  sequence_log_prob(logits, tokens, mask)      log-probability of token sequences (differentiable)
Search
  mcts(state, legal_actions, apply, is_terminal, reward, simulations, c, rollout_depth, seed)
  beam_search(step, start, beam_width, max_len, eos)
  best_of_n(candidates, score)
"""
import math

import numpy as np

from . import ops
from .errors import RuntimeErr
from .units import Quantity
from .values import Record, tree_map


def _v(x):
    if isinstance(x, Quantity):
        x = x.v
    return np.asarray(ops.raw(x))


def _rng(seed):
    return np.random.default_rng(None if seed is None else int(seed))


# =============================================================================== environments
class CartPole:
    """Vectorised CartPole-v1: 4 observations, 2 actions, reward 1 per step, episodes end at |θ| > 12°, |x| > 2.4
    or 500 steps."""
    obs_dim, n_actions = 4, 2

    def __init__(self, n=1, seed=None, max_steps=500):
        self.n, self.max_steps = int(n), int(max_steps)
        self.rng = _rng(seed)
        self.g, self.mc, self.mp, self.l, self.f, self.tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        self.state = np.zeros((self.n, 4))
        self.t = np.zeros(self.n, dtype=np.int64)
        self.returns = np.zeros(self.n)
        self.episode_returns = []

    def reset(self, seed=None):
        if seed is not None:
            self.rng = _rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, (self.n, 4))
        self.t[:] = 0
        self.returns[:] = 0
        return self.state.copy()

    def step(self, actions):
        a = _v(actions).astype(np.int64).reshape(self.n)
        x, xd, th, thd = self.state.T
        force = np.where(a == 1, self.f, -self.f)
        ct, st = np.cos(th), np.sin(th)
        total = self.mc + self.mp
        pml = self.mp * self.l
        temp = (force + pml * thd ** 2 * st) / total
        thacc = (self.g * st - ct * temp) / (self.l * (4.0 / 3.0 - self.mp * ct ** 2 / total))
        xacc = temp - pml * thacc * ct / total
        x = x + self.tau * xd
        xd = xd + self.tau * xacc
        th = th + self.tau * thd
        thd = thd + self.tau * thacc
        self.state = np.stack([x, xd, th, thd], 1)
        self.t += 1
        done = (np.abs(x) > 2.4) | (np.abs(th) > 12 * 2 * math.pi / 360)
        trunc = (self.t >= self.max_steps) & ~done
        reward = np.ones(self.n)
        self.returns += reward
        final = self.state.copy()
        end = done | trunc
        if end.any():
            self.episode_returns.extend(self.returns[end].tolist())
            self.state[end] = self.rng.uniform(-0.05, 0.05, (int(end.sum()), 4))
            self.t[end] = 0
            self.returns[end] = 0
        return Record(obs=self.state.copy(), reward=reward, done=done.astype(float), truncated=trunc.astype(float),
                      final_obs=final)


class GridWorld:
    """Reach the bottom-right corner of a size×size grid. Observation: one-hot position. Actions: up, down,
    left, right. Reward −0.01 per step and +1 at the goal; episodes end at the goal or after 4·size² steps."""

    def __init__(self, size=5, n=1, seed=None):
        self.size, self.n = int(size), int(n)
        self.obs_dim, self.n_actions = self.size * self.size, 4
        self.rng = _rng(seed)
        self.pos = np.zeros((self.n, 2), dtype=np.int64)
        self.t = np.zeros(self.n, dtype=np.int64)
        self.returns = np.zeros(self.n)
        self.episode_returns = []

    def _obs(self):
        o = np.zeros((self.n, self.obs_dim))
        o[np.arange(self.n), self.pos[:, 0] * self.size + self.pos[:, 1]] = 1.0
        return o

    def reset(self, seed=None):
        if seed is not None:
            self.rng = _rng(seed)
        self.pos[:] = 0
        self.t[:] = 0
        self.returns[:] = 0
        return self._obs()

    def step(self, actions):
        a = _v(actions).astype(np.int64).reshape(self.n)
        d = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])[a]
        self.pos = np.clip(self.pos + d, 0, self.size - 1)
        self.t += 1
        goal = (self.pos[:, 0] == self.size - 1) & (self.pos[:, 1] == self.size - 1)
        reward = np.where(goal, 1.0, -0.01)
        trunc = (self.t >= 4 * self.size * self.size) & ~goal
        self.returns += reward
        final = self._obs()
        end = goal | trunc
        if end.any():
            self.episode_returns.extend(self.returns[end].tolist())
            self.pos[end] = 0
            self.t[end] = 0
            self.returns[end] = 0
        return Record(obs=self._obs(), reward=reward, done=goal.astype(float), truncated=trunc.astype(float),
                      final_obs=final)


class Bandit:
    """k-armed Gaussian bandit; the observation is constant. Arm means come from N(0, 1)."""

    def __init__(self, arms=10, n=1, seed=None):
        self.k, self.n = int(arms), int(n)
        self.rng = _rng(seed)
        self.means = self.rng.standard_normal(self.k)
        self.obs_dim, self.n_actions = 1, self.k
        self.episode_returns = []

    def reset(self, seed=None):
        return np.ones((self.n, 1))

    def step(self, actions):
        a = _v(actions).astype(np.int64).reshape(self.n)
        r = self.means[a] + self.rng.standard_normal(self.n)
        self.episode_returns.extend(r.tolist())
        z = np.zeros(self.n)
        return Record(obs=np.ones((self.n, 1)), reward=r, done=np.ones(self.n), truncated=z, final_obs=np.ones((self.n, 1)))


def cartpole(n=1, seed=None, max_steps=500):
    return CartPole(n, seed, max_steps)


def gridworld(size=5, n=1, seed=None):
    return GridWorld(size, n, seed)


def bandit(arms=10, n=1, seed=None):
    return Bandit(arms, n, seed)


# =============================================================================== returns and advantages
def discounted_returns(rewards, dones, gamma=0.99, last_value=None):
    """Discounted returns of time-major rewards [T] or [T, N]; done[t] ends the episode after step t."""
    r, d = _v(rewards).astype(float), _v(dones).astype(float)
    out = np.zeros_like(r)
    run = np.zeros(r.shape[1:]) if last_value is None else _v(last_value).astype(float).copy()
    for t in range(r.shape[0] - 1, -1, -1):
        run = r[t] + gamma * run * (1.0 - d[t])
        out[t] = run
    return out


def gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation (time-major [T, N]). Returns record(advantages, returns)."""
    r, v, d = _v(rewards).astype(float), _v(values).astype(float), _v(dones).astype(float)
    nv = _v(last_values).astype(float)
    adv = np.zeros_like(r)
    last = np.zeros(r.shape[1:])
    for t in range(r.shape[0] - 1, -1, -1):
        next_v = nv if t == r.shape[0] - 1 else v[t + 1]
        nonterm = 1.0 - d[t]
        delta = r[t] + gamma * next_v * nonterm - v[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return Record(advantages=adv, returns=adv + v)


# =============================================================================== distributions
def categorical_log_prob(logits, actions):
    """log π(a) for integer actions (differentiable in logits)."""
    lp = ops.log_softmax(logits, axis=-1)
    a = _v(actions).astype(np.int64)
    idx = np.arange(a.shape[0])
    return lp[(idx, a)]


def categorical_entropy(logits):
    lp = ops.log_softmax(logits, axis=-1)
    return -ops.sum_(ops.exp(lp) * lp, axis=-1)


def categorical_sample(logits, seed=None, rng=None):
    """Sample actions from softmax(logits) (Gumbel-max)."""
    lg = _v(logits).astype(float)
    g = (rng or _rng(seed)).gumbel(size=lg.shape)
    return np.argmax(lg + g, axis=-1)


def gaussian_log_prob(mean, log_std, actions):
    """Sum over action dimensions of log N(a; mean, exp(log_std)²) (differentiable)."""
    a = actions
    z = (a - mean) / ops.exp(log_std)
    per = -0.5 * z * z - log_std - 0.5 * math.log(2 * math.pi)
    return ops.sum_(per, axis=-1)


def gaussian_entropy(log_std):
    return ops.sum_(log_std + 0.5 * math.log(2 * math.pi * math.e), axis=-1)


def sample_token(logits, temperature=1.0, top_k=None, top_p=None, seed=None):
    """Sample from logits [..., V] with temperature, top-k and nucleus (top-p) filtering; temperature 0 = greedy."""
    lg = _v(logits).astype(float)
    if temperature == 0:
        return np.argmax(lg, axis=-1)
    lg = lg / float(temperature)
    if top_k:
        k = int(top_k)
        kth = np.sort(lg, axis=-1)[..., -k][..., None]
        lg = np.where(lg < kth, -np.inf, lg)
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-lg, axis=-1)
        sl = np.take_along_axis(lg, order, -1)
        pr = np.exp(sl - sl.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        cum = np.cumsum(pr, -1)
        cut = cum - pr > float(top_p)
        sl = np.where(cut, -np.inf, sl)
        lg = np.full_like(lg, -np.inf)
        np.put_along_axis(lg, order, sl, -1)
    return categorical_sample(lg, seed)


# =============================================================================== algorithms
def _split(out):
    if isinstance(out, (list, tuple)) and len(out) == 2:
        return out[0], out[1]
    if isinstance(out, Record):
        return out.logits, out.value
    raise RuntimeErr("the model must return [logits, value]")


def _clip(grads, max_norm):
    from .torch_compat import clip_grad_norm
    return clip_grad_norm(grads, max_norm) if max_norm else grads


def ppo(env, model, params, total_steps=100000, rollout=128, epochs=4, minibatch=256, lr=3e-4, gamma=0.99,
        lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, normalize_advantages=True,
        anneal_lr=True, seed=0, verbose=False, opt=None):
    """Proximal Policy Optimization (clipped surrogate, GAE, value and entropy terms, gradient clipping).
    `model(params, obs)` returns [logits, value]. Returns record(params, episode_returns, updates)."""
    from .stdlib import value_and_grad
    from .torch_compat import Adam
    rng = _rng(seed)
    n = env.n
    obs = env.reset(seed)
    opt = opt or Adam(lr=lr, eps=1e-5)
    updates = max(1, int(total_steps) // (rollout * n))
    log = []
    for u in range(updates):
        if anneal_lr:
            opt.lr = lr * (1.0 - u / updates)
        O, A, LP, V, R, D = [], [], [], [], [], []
        for t in range(rollout):
            logits, value = _split(model(params, obs))
            lg = np.asarray(ops.raw(logits))
            a = categorical_sample(lg, rng=rng)
            lp = lg - lg.max(-1, keepdims=True)
            lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
            r = env.step(a)
            O.append(obs)
            A.append(a)
            LP.append(lp[np.arange(n), a])
            V.append(np.asarray(ops.raw(value)).reshape(n))
            rew = r.reward.copy()
            if r.truncated.any():                         # bootstrap from the true final state when truncated
                _, fv = _split(model(params, r.final_obs))
                rew = rew + gamma * np.asarray(ops.raw(fv)).reshape(n) * r.truncated
            R.append(rew)
            D.append(np.maximum(r.done, r.truncated))
            obs = r.obs
        _, last_v = _split(model(params, obs))
        g = gae(np.array(R), np.array(V), np.array(D), np.asarray(ops.raw(last_v)).reshape(n), gamma, lam)
        bO = np.concatenate(O)
        bA = np.concatenate(A)
        bLP = np.concatenate(LP)
        bAdv = g.advantages.reshape(-1)
        bRet = g.returns.reshape(-1)
        N = bO.shape[0]
        for _ in range(int(epochs)):
            perm = rng.permutation(N)
            for s in range(0, N, int(minibatch)):
                idx = perm[s:s + int(minibatch)]
                adv = bAdv[idx]
                if normalize_advantages and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                def loss(p):
                    logits, value = _split(model(p, bO[idx]))
                    nlp = categorical_log_prob(logits, bA[idx])
                    ratio = ops.exp(nlp - bLP[idx])
                    s1 = ratio * adv
                    s2 = ops.clip(ratio, 1 - clip, 1 + clip) * adv
                    pg = -ops.mean(ops.where(np.asarray(ops.raw(s1)) < np.asarray(ops.raw(s2)), s1, s2))
                    vl = ops.mean((ops.reshape(value, -1) - bRet[idx]) ** 2)
                    ent = ops.mean(categorical_entropy(logits))
                    return pg + vf_coef * vl - ent_coef * ent
                _, gr = value_and_grad(loss)(params)
                params = opt.step(params, _clip(gr, max_norm_value(max_grad_norm)))
        recent = env.episode_returns[-20:]
        log.append(float(np.mean(recent)) if recent else 0.0)
        if verbose and (u % max(1, updates // 10) == 0 or u == updates - 1):
            print(f"update {u + 1}/{updates}: mean return {log[-1]:.1f}")
    return Record(params=params, episode_returns=list(env.episode_returns), mean_returns=log, updates=updates)


def max_norm_value(v):
    return float(v) if v else None


def dqn(env, model, params, total_steps=50000, buffer_size=50000, batch_size=64, lr=1e-3, gamma=0.99,
        learning_starts=1000, train_every=1, target_every=500, eps_start=1.0, eps_end=0.05, eps_fraction=0.3,
        double=True, max_grad_norm=10.0, seed=0, verbose=False):
    """Deep Q-learning: replay buffer, ε-greedy exploration, target network, (double) DQN target, Huber loss.
    model(params, obs) returns Q-values [batch, actions]. Returns record(params, episode_returns)."""
    from .stdlib import value_and_grad
    from .torch_compat import Adam, huber_loss
    rng = _rng(seed)
    n = env.n
    obs = env.reset(seed)
    od = obs.shape[1:]
    B = int(buffer_size)
    bo, bn = np.zeros((B,) + od), np.zeros((B,) + od)
    ba, br, bd = np.zeros(B, dtype=np.int64), np.zeros(B), np.zeros(B)
    size = ptr = 0
    target = params
    opt = Adam(lr=lr)
    steps = int(total_steps) // n
    for t in range(steps):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - t / max(1, eps_fraction * steps))
        q = np.asarray(ops.raw(model(params, obs)))
        a = np.where(rng.random(n) < eps, rng.integers(0, q.shape[1], n), q.argmax(1))
        r = env.step(a)
        for i in range(n):
            bo[ptr], ba[ptr], br[ptr], bn[ptr], bd[ptr] = obs[i], a[i], r.reward[i], r.final_obs[i], r.done[i]
            ptr = (ptr + 1) % B
            size = min(size + 1, B)
        obs = r.obs
        if size >= learning_starts and t % train_every == 0:
            idx = rng.integers(0, size, batch_size)
            qn_t = np.asarray(ops.raw(model(target, bn[idx])))
            if double:
                an = np.asarray(ops.raw(model(params, bn[idx]))).argmax(1)
                qn = qn_t[np.arange(batch_size), an]
            else:
                qn = qn_t.max(1)
            y = br[idx] + gamma * (1 - bd[idx]) * qn

            def loss(p):
                qs = model(p, bo[idx])
                return huber_loss(qs[(np.arange(batch_size), ba[idx])], y)
            _, gr = value_and_grad(loss)(params)
            params = opt.step(params, _clip(gr, max_grad_norm))
        if t % target_every == 0:
            target = params
        if verbose and t % max(1, steps // 10) == 0:
            rec = env.episode_returns[-20:]
            print(f"step {t * n}: mean return {np.mean(rec) if rec else 0:.2f}  ε={eps:.2f}")
    return Record(params=params, episode_returns=list(env.episode_returns))


def reinforce(env, model, params, episodes=500, lr=1e-2, gamma=0.99, seed=0, max_steps=1000):
    """REINFORCE with a running-mean baseline on a single environment (n=1). model(params, obs) -> logits."""
    from .stdlib import value_and_grad
    from .torch_compat import Adam
    rng = _rng(seed)
    opt = Adam(lr=lr)
    baseline = 0.0
    totals = []
    for ep in range(int(episodes)):
        obs = env.reset(None if ep else seed)
        O, A, R = [], [], []
        for _ in range(int(max_steps)):
            lg = np.asarray(ops.raw(model(params, obs)))
            a = categorical_sample(lg, rng=rng)
            r = env.step(a)
            O.append(obs[0])
            A.append(int(a[0]))
            R.append(float(r.reward[0]))
            obs = r.obs
            if r.done[0] or r.truncated[0]:
                break
        G = discounted_returns(np.array(R), np.zeros(len(R)), gamma)
        baseline = 0.9 * baseline + 0.1 * float(G.mean())
        adv = G - baseline
        Oa, Aa = np.array(O), np.array(A)

        def loss(p):
            return -ops.mean(categorical_log_prob(model(p, Oa), Aa) * adv)
        _, gr = value_and_grad(loss)(params)
        params = opt.step(params, gr)
        totals.append(sum(R))
    return Record(params=params, episode_returns=totals)


def evaluate(env, policy, episodes=10, seed=0):
    """Run policy(obs) -> actions until `episodes` episodes have finished; returns their mean and all returns."""
    env.episode_returns = []
    obs = env.reset(seed)
    guard = 0
    while len(env.episode_returns) < episodes and guard < 10 ** 7:
        obs = env.step(_v(policy(obs))).obs
        guard += 1
    rets = env.episode_returns[:episodes]
    return Record(mean=float(np.mean(rets)) if rets else 0.0, returns=rets)


# =============================================================================== learning from preferences
def _log_sigmoid(x):
    return -ops.softplus(-x)


def preference_loss(r_chosen, r_rejected, margin=0.0):
    """Bradley–Terry reward-model loss: −log σ(r_chosen − r_rejected − margin), averaged."""
    return -ops.mean(_log_sigmoid(r_chosen - r_rejected - margin))


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, label_smoothing=0.0):
    """Direct Preference Optimization. Inputs are summed log-probabilities of the chosen / rejected responses under
    the policy and the frozen reference model. Returns record(loss, chosen_rewards, rejected_rewards, accuracy)."""
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    ls = float(label_smoothing)
    loss = -ops.mean(_log_sigmoid(beta * logits) * (1 - ls) + _log_sigmoid(-beta * logits) * ls)
    cr = beta * (np.asarray(ops.raw(policy_chosen)) - np.asarray(ops.raw(ref_chosen)))
    rr = beta * (np.asarray(ops.raw(policy_rejected)) - np.asarray(ops.raw(ref_rejected)))
    return Record(loss=loss, chosen_rewards=cr, rejected_rewards=rr, accuracy=float(np.mean(cr > rr)))


def ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau=0.1):
    """Identity Preference Optimization: mean((h − 1/(2τ))²) with h the log-ratio difference."""
    h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    d = h - 1.0 / (2.0 * tau)
    return ops.mean(d * d)


def kl_penalized_rewards(rewards, logp, ref_logp, beta=0.05):
    """Per-token rewards for RLHF with PPO: −β·(log π − log π_ref) at every token plus the sequence reward on the
    last token. logp / ref_logp: [B, T]; rewards: [B]. Returns [B, T]."""
    lp, rp = _v(logp).astype(float), _v(ref_logp).astype(float)
    out = -float(beta) * (lp - rp)
    out[:, -1] += _v(rewards).astype(float)
    return out


def grpo_advantages(rewards, group_size, eps=1e-6):
    """GRPO: rewards [B·G] of G completions per prompt -> (r − group mean) / (group std + eps)."""
    r = _v(rewards).astype(float).reshape(-1, int(group_size))
    a = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + eps)
    return a.reshape(-1)


def ppo_clip_loss(logp, old_logp, advantages, clip=0.2, mask=None):
    """Clipped PPO surrogate for token log-probs [B, T] (mask: 1 for response tokens)."""
    ratio = ops.exp(logp - old_logp)
    adv = advantages
    s1 = ratio * adv
    s2 = ops.clip(ratio, 1 - clip, 1 + clip) * adv
    obj = ops.where(np.asarray(ops.raw(s1)) < np.asarray(ops.raw(s2)), s1, s2)
    if mask is None:
        return -ops.mean(obj)
    m = _v(mask).astype(float)
    return -ops.sum_(obj * m) / max(m.sum(), 1.0)


def sequence_log_prob(logits, tokens, mask=None):
    """Σ_t log softmax(logits[b, t])[tokens[b, t]] for each sequence (logits [B, T, V], tokens [B, T])."""
    lp = ops.log_softmax(logits, axis=-1)
    tk = _v(tokens).astype(np.int64)
    B, T = tk.shape
    picked = lp[(np.arange(B)[:, None], np.arange(T)[None, :], tk)]
    if mask is not None:
        picked = picked * _v(mask).astype(float)
    return ops.sum_(picked, axis=-1)


# =============================================================================== search
class _Node:
    __slots__ = ("state", "parent", "action", "children", "untried", "visits", "value", "player")

    def __init__(self, state, parent, action, untried):
        self.state, self.parent, self.action = state, parent, action
        self.children, self.untried = [], list(untried)
        self.visits, self.value = 0, 0.0


def mcts(state, legal_actions, apply, is_terminal, reward, simulations=1000, c=1.41, rollout_depth=200, seed=0,
         two_player=False):
    """Monte-Carlo tree search (UCT). Functions: legal_actions(s) -> list, apply(s, a) -> new state,
    is_terminal(s) -> bool, reward(s) -> value from the point of view of the player who moved last when
    two_player=true (otherwise of the single agent). Returns record(action, visits, values)."""
    rng = _rng(seed)
    root = _Node(state, None, None, legal_actions(state))
    for _ in range(int(simulations)):
        node = root
        while not node.untried and node.children:
            lg = math.log(node.visits)
            node = max(node.children, key=lambda ch: ch.value / ch.visits + c * math.sqrt(lg / ch.visits))
        if node.untried and not is_terminal(node.state):
            a = node.untried.pop(int(rng.integers(len(node.untried))))
            s2 = apply(node.state, a)
            child = _Node(s2, node, a, [] if is_terminal(s2) else legal_actions(s2))
            node.children.append(child)
            node = child
        s = node.state
        depth = 0
        flips = 0
        while not is_terminal(s) and depth < rollout_depth:
            acts = legal_actions(s)
            s = apply(s, acts[int(rng.integers(len(acts)))])
            depth += 1
            flips += 1
        v = float(reward(s))
        if two_player and flips % 2 == 1:
            v = -v
        while node is not None:
            node.visits += 1
            node.value += v
            if two_player:
                v = -v
            node = node.parent
    best = max(root.children, key=lambda ch: ch.visits)
    return Record(action=best.action, visits={str(ch.action): ch.visits for ch in root.children},
                  values={str(ch.action): ch.value / ch.visits for ch in root.children})


def beam_search(step, start, beam_width=4, max_len=20, eos=None):
    """Beam search: step(sequence) returns log-probabilities over the next token (1-D). Returns the best sequences
    as a list of record(tokens, score), best first."""
    beams = [(list(start), 0.0, False)]
    for _ in range(int(max_len)):
        cand = []
        for seq, score, done in beams:
            if done:
                cand.append((seq, score, True))
                continue
            lp = _v(step(seq)).astype(float).reshape(-1)
            top = np.argsort(-lp)[:beam_width]
            for t in top:
                cand.append((seq + [int(t)], score + float(lp[t]), eos is not None and int(t) == int(eos)))
        cand.sort(key=lambda b: -b[1])
        beams = cand[:beam_width]
        if all(b[2] for b in beams):
            break
    return [Record(tokens=b[0], score=b[1]) for b in beams]


def best_of_n(candidates, score):
    """The candidate with the highest score(candidate), with all scores."""
    sc = [float(_v(score(c))) for c in candidates]
    i = int(np.argmax(sc))
    return Record(best=candidates[i], index=i, scores=sc)


def module():
    return {k: v for k, v in globals().items()
            if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__
            and k not in ("module", "max_norm_value") and not isinstance(v, type)}
