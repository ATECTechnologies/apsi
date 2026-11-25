# Physics-informed networks

A physics-informed network learns the solution of an equation from the equation itself: the loss is the equation's
residual at chosen points, written with the network's derivatives, plus the boundary and initial conditions and any
measurements you have. `use layers` gives two functions for it.

```apsi
use layers
use nn

# the heat equation u_t = a u_xx, u(x, 0) = sin(pi x), u(0, t) = u(1, t) = 0
fn heat_loss(x, u, du, d2u):
    let res = du[:n_in, 1, 0] - a * d2u[:n_in, 0, 0]          # inside: u_t - a u_xx
    let ic = u[n_in:n_in + n_ic, 0] - sin(pi * x[n_in:n_in + n_ic, 0])
    let bc = u[n_in + n_ic:, 0]
    return mean(res ** 2) + 10.0 * mean(ic ** 2) + 10.0 * mean(bc ** 2)

let net = mlp([2, 32, 32, 1], "tanh")
let r = pinn_fit(net, net.init(seed=0), X, heat_loss, epochs=1500, opt=adam(lr=0.005))
```

| | |
|---|---|
| `pinn_fit(model, params, points, loss, epochs=2000, opt=adam(1e-3), backend="auto", device="cuda", schedule=, clip=, every=, compensated=true)` | trains; `loss(x, u, du, d2u)` returns a number. Returns `params`, `history`, `backend`, `seconds` |
| `pinn_derivatives(model, params, points)` | `u`, `du`, `d2u` at the points |
| `u` | `[N, m]`: the network's m outputs at the N points |
| `du` | `[N, d, m]`: `du[:, i, k]` = ∂u_k / ∂x_i |
| `d2u` | `[N, d, m]`: `d2u[:, i, k]` = ∂²u_k / ∂x_i² (sum them over i for a Laplacian) |

Everything the equation needs goes into one set of points: the inside, the boundary, the initial line, the
measurements; the loss picks each part by its rows (or by the values of `x`). Any formula of `x`, `u`, `du`, `d2u`
works: systems of equations (several outputs), coefficients that depend on `x`, non-linear terms, weights between
the parts. `examples/44_physics_networks.apsi` learns the heat equation (error 0.001 against the exact solution) and
a damped oscillator with `sin` activations.

## `use pinn`: the same thing in a few lines, natively too

`use pinn` is written in Apsi and is the same file for `apsi run` and for native programs (`apsi build --native`).
A network is one object: build it, train it with `fit`, ask it with `u` or `derivatives`.

```apsi
use pinn

# u'' + 4u = 0, u(0) = 1, u'(0) = 0
let net = pinn.mlp([1, 24, 24, 1], act="tanh")
let t = reshape(linspace(0.0, 2.0, 40), 40, 1)
fn loss(x, u, du, d2u):
    return [mean((d2u[:, 0, 0] + 4.0 * u[:, 0]) ** 2),           # the equation
            (u[0, 0] - 1.0) ** 2 + du[0, 0, 0] ** 2]              # the initial conditions
let r = net.fit(t, loss, epochs=1500, lr=1e-2, adapt=true)
print(net.u(t))
```

- **Derivatives in one pass.** Every layer carries the values and their first and second derivatives along each
  input in one stacked matrix product (Taylor mode). `u [N, m]`, `du [N, d, m]`, `d2u [N, d, m]`; with
  `hessian=true` the loss also gets `hess [N, d, d, m]` with the mixed derivatives (elasticity, cross-diffusion,
  Navier–Stokes in primitive variables). They agree with nested `grad` to rounding.
- **Networks.** `act=` `tanh`, `sin`, `cos`, `sigmoid`, `softplus`, `silu`, `linear`; `fourier=F, sigma=s` maps the
  inputs to `sin` / `cos` of `2πxB` first (B random, fixed), so a network learns fine detail and high frequencies;
  any number of outputs for systems of equations; `dtype="float64"` by default.
- **A loss of several terms.** Return a list (equation, boundaries, initial conditions, data). `adapt=true` sets
  their weights from the sizes of their gradients (gradient-norm balancing) every `adapt_every` steps, so no term
  drowns the others; the weights are in `r.weights` and kept for the next `fit`.
- **Inverse problems.** `pinn.mlp(..., unknowns={"nu": 0.05})` learns physical constants with the network; the loss
  gets them as a last argument `k` (`k["nu"]`), `r.k` / `net.k()` read them. They learn `k_lr` times faster than the
  weights (default 10).
- **Points.** `pinn.box(lo, hi, n)` inside a box, `pinn.faces(lo, hi, n)` on its boundary, `pinn.grid(lo, hi, n)`;
  `points=fn(step) => …` draws new collocation points every `resample_every` steps.
- **Training.** Adam with warm-up and a cosine schedule down to `min_lr`, `clip=` for the gradient norm, `every=`
  prints the loss. Helpers for equations: `pinn.laplacian(d2u)`, `pinn.divergence(du, outs)`, `pinn.mse(r)`.
- **Your own parts.** `pinn.taylor(net, params, x, hessian)` and `pinn.act3(name, z)` are plain functions: copy
  one, change it (another activation, another layer), and the rest keeps working.

## On the GPU and on the processor

`backend="auto"` takes the GPU when there is one and the network is made of linear layers and smooth activations
(`tanh`, `sin`, `cos`, `sigmoid`, `silu`, `softplus`); otherwise the processor.

- **GPU, Taylor mode.** Every layer carries the values, their first derivatives and their second derivatives along
  each input in one stack: a linear layer is one matrix product for all of them, an activation one kernel that applies
  f, f′ and f″ (f(z), f′(z) j, f″(z) j² + f′(z) h). The gradient flows back through the same stack (a kernel with f′,
  f″ and f‴). Only u, du, d2u and their gradients go to the processor, where the loss is evaluated with Apsi's
  autodiff, so the loss can be any formula. The weights are updated on the card with compensated summation.
- **Processor.** The derivatives come from Apsi's autodiff of any order (`grad` of `grad`), for any model,
  including graph models (`graph=`), custom layers and third or mixed derivatives if you write the loss with `grad`
  yourself.

Both give the same training: the tests compare them step by step (to float32 precision).

Mixed second derivatives (∂²u/∂x∂y) are not in `d2u`; write that loss with `grad` on the processor.

## Getting a good solution

- **Weigh the conditions.** Boundary and initial conditions usually need a larger weight than the residual (10 in
  the example): without it the network satisfies the equation with the wrong boundary values.
- **Scale the inputs** to about [-1, 1] or [0, 1]. The derivatives are taken with respect to what the network sees;
  if you scale `x` by `s`, multiply `du` by `s` and `d2u` by `s²` in the loss.
- **Smooth activations.** `tanh` for most problems, `sin` for oscillating solutions (the damped oscillator in the
  example). ReLU's second derivative is zero almost everywhere, so it cannot learn second-order equations.
- **Enough points where the solution changes fast**: more collocation points near steep fronts and boundaries.
- **Learning rate:** `adam(lr=5e-3)` to start, lower (1e-3 → 1e-4, or `schedule=`) once the loss flattens.
- **Check against what you know**: an exact solution, a conserved quantity or measurements, at points that were not
  in the training set.

## How it was checked

The tests train the same network on the GPU (Taylor mode) and on the processor (autodiff) and compare every step to
float32 precision; the heat equation of the example reaches an error of about 0.001 against the exact solution.
`apsi gpu bench check` runs the physics-informed network on your card against the reference.
