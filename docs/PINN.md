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
