# Reinforcement learning

`use rl` provides environments, advantage estimation, policy helpers, complete training algorithms (PPO, DQN, REINFORCE), the losses used to train language models from human preferences (RLHF, DPO, IPO, GRPO), and search algorithms (Monte Carlo tree search, beam search, best-of-n). Everything runs on the CPU without external libraries and works with any model written with `use torch` or `use layers`.

## A complete example: PPO on CartPole

```apsi
use torch
use rl

let l1 = torch.Linear(4, 64)
let l2 = torch.Linear(64, 64)
let pi = torch.Linear(64, 2)
let vf = torch.Linear(64, 1)
let params = record(l1=l1.init(1), l2=l2.init(2), pi=pi.init(3), vf=vf.init(4))

# the model returns [action logits, state value]
fn policy(p, obs):
    let h = tanh(l2(p.l2, tanh(l1(p.l1, obs))))
    return [pi(p.pi, h), vf(p.vf, h)]

let result = rl.ppo(rl.cartpole(8, seed=0), policy, params, total_steps=200000, epochs=10, lr=1e-3)
let score = rl.evaluate(rl.cartpole(1, seed=5), fn(obs) => argmax(policy(result.params, obs)[0], axis=1), episodes=5)
print(f"mean return: {score.mean}")        # 500 is the maximum
```

With these settings PPO reaches the maximum return of 500 on CartPole-v1 in about 200 000 environment steps.

## Environments

| Function | Task |
|---|---|
| `cartpole(n=1, seed, max_steps=500)` | Balance a pole on a cart. Dynamics identical to Gymnasium’s CartPole-v1. |
| `gridworld(size=5, n=1, seed)` | Walk to the goal on a grid; each step costs a little reward. |
| `bandit(arms=10, n=1, seed)` | Multi-armed Gaussian bandit. |

All environments are **vectorised**: `n` copies run side by side. `reset(seed)` returns observations of shape `[n, …]`; `step(actions)` returns `record(obs, reward, done, truncated, final_obs)`. Finished copies reset themselves, and `final_obs` holds the last observation before the reset. Completed episode returns are collected in `env.episode_returns`.

Your own environment only needs the same members: `n`, `reset(seed)`, `step(actions)` and `episode_returns`.

## Algorithms

### `ppo(env, model, params, …)`

Proximal Policy Optimization with the clipped surrogate objective, generalised advantage estimation, a value loss, an optional entropy bonus, advantage normalisation, global gradient-norm clipping and linear learning-rate annealing.

| Argument | Default | Meaning |
|---|---|---|
| `total_steps` | 100 000 | environment steps across all copies |
| `rollout` | 128 | steps per copy between updates |
| `epochs` | 4 | passes over each rollout |
| `minibatch` | 256 | samples per gradient step |
| `lr` | 3e-4 | Adam learning rate |
| `gamma`, `lam` | 0.99, 0.95 | discount and GAE parameter |
| `clip` | 0.2 | ratio clipping range |
| `vf_coef`, `ent_coef` | 0.5, 0.0 | weights of the value loss and entropy bonus |
| `max_grad_norm` | 0.5 | gradient clipping |
| `anneal_lr` | true | decay the learning rate linearly to 0 |
| `opt` | Adam | any optimizer from `use torch` |

`model(params, obs)` returns `[logits, value]`. The result is `record(params, episode_returns, mean_returns, updates)`.

### `dqn(env, model, params, …)`

Double DQN with a replay buffer, a target network, ε-greedy exploration and the Huber loss. `model(params, obs)` returns one value per action.

### `reinforce(env, model, params, episodes, …)`

Monte Carlo policy gradient with a running baseline.

### `evaluate(env, policy, episodes=10, seed=0)`

Runs a deterministic policy (`policy(obs)` returns actions) and returns `record(mean, returns)`.

## Building blocks

| Function | Purpose |
|---|---|
| `discounted_returns(rewards, dones, gamma, last_value)` | discounted returns that stop at episode ends |
| `gae(rewards, values, dones, last_values, gamma, lam)` | generalised advantage estimation; returns `record(advantages, returns)` |
| `categorical_sample(logits, seed)` · `categorical_log_prob(logits, actions)` · `categorical_entropy(logits)` | discrete policies |
| `gaussian_log_prob(mean, log_std, actions)` · `gaussian_entropy(log_std)` | continuous policies |
| `sample_token(logits, temperature, top_k, top_p, seed)` | sampling for language models |

## Learning from preferences (RLHF, DPO, GRPO)

| Function | Purpose |
|---|---|
| `preference_loss(r_chosen, r_rejected, margin)` | Bradley–Terry loss for training a reward model |
| `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, label_smoothing)` | Direct Preference Optimization; inputs are sequence log-probabilities; returns `record(loss, chosen_rewards, rejected_rewards, accuracy)` |
| `ipo_loss(…, tau)` | Identity Preference Optimization |
| `kl_penalized_rewards(rewards, logp, ref_logp, beta)` | per-token rewards with a KL penalty against a reference model (RLHF with PPO) |
| `grpo_advantages(rewards, group_size)` | group-relative advantages: several answers per prompt, normalised within each group |
| `ppo_clip_loss(logp, old_logp, advantages, clip, mask)` | the clipped surrogate for token sequences |
| `sequence_log_prob(logits, tokens, mask)` | log-probability of each sequence, differentiable |

A DPO step for a causal language model:

```apsi
fn dpo_step(p):
    let pc = rl.sequence_log_prob(model(p, chosen), chosen_targets, chosen_mask)
    let pr = rl.sequence_log_prob(model(p, rejected), rejected_targets, rejected_mask)
    return rl.dpo_loss(pc, pr, ref_chosen, ref_rejected, beta=0.1).loss

let loss, grads = value_and_grad(dpo_step)(params)
params = opt.step(params, grads)
```

`ref_chosen` and `ref_rejected` are the same log-probabilities computed once with the frozen reference model.

## Search

| Function | Purpose |
|---|---|
| `mcts(state, legal_actions, apply, is_terminal, reward, simulations, c, rollout_depth, seed, two_player)` | Monte Carlo tree search with UCT; returns `record(action, visits, values)` |
| `beam_search(step, start, beam_width, max_len, eos)` | keeps the best partial sequences; `step(seq)` returns log-probabilities of the next token; returns a list of `record(tokens, score)`, best first |
| `best_of_n(candidates, score)` | the highest-scoring candidate (rejection sampling); returns `record(best, index, scores)` |

## Memory and speed

- Train in float32 with `fit(…, precision="fp32")` or by creating parameters with `dtype=f32`; gradients and optimizer state stay float32, which halves memory.
- Wrap expensive blocks with `torch.checkpoint_layer(layer, params, x)` to recompute activations in the backward pass instead of storing them: roughly a third of the peak memory for about twice the compute.
- Run many environment copies (`n=8` or more) so each model call processes a batch.
