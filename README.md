# Stiff Beam Problem

A voluntary exercise to deepen understanding of numerical methods for stiff ODEs covered in lectures as part of Master's degree. In line with this purpose, the code was written without the aid of AI to be as simple and clear as possible so it is not designed to be efficient!

---

## The Problem

A clamped elastic beam of length L is described by its deflection angle θ(s, t), where s ∈ [0, 1] is the arc length and t is time. A force is applied at the free tip for the first π time units:

$$
F(t) = \begin{pmatrix} -\alpha(t) \\ \alpha(t) \end{pmatrix}, \qquad \alpha(t) = \begin{cases} 1.5\sin^2(t) & 0 \leq t \leq \pi \\ 0 & t > \pi \end{cases}
$$

After spatial semi-discretisation into S segments, the deflection angles θ₁, …, θ_S satisfy the coupled second-order ODE system

$$
\ddot{\theta} = Cv + DC^{-1}(Dv + \dot{\theta}^2)
$$

where C and D are angle-dependent matrices derived from the spatial discretisation, and v encodes the bending stiffness and external force. Introducing the angular velocity κ = θ̇, this is rewritten as a first-order system in (θ, κ) and solved using Nyström-adapted Runge–Kutta methods.

---

## Code Structure

```
├── main.py               # Problem setup, RKM time-stepping, demo mode, and 3D plotting
├── fkappa.py             # Equation of motion: computes κ̇ = f_κ(t, θ, κ)
├── compute_C.py          # Builds the angle-dependent matrix C(θ)
├── compute_D.py          # Builds the angle-dependent matrix D(θ)
├── compute_v.py          # Computes the forcing/stiffness vector v(t, θ)
├── compute_J.py          # Jacobian of f_κ w.r.t. θ (Jθ) and κ (Jκ), for Newton iteration
├── compute_Ctheta.py     # Tensor of partial derivatives ∂C/∂θ
├── compute_Dtheta.py     # Tensor of partial derivatives ∂D/∂θ
└── compute_vtheta.py     # Matrix of partial derivatives ∂v/∂θ
```

The core computation lives in `main.py` inside the `stiffbeam()` function. The helper modules each handle one piece of the mathematical machinery, keeping the time-stepping loop clean.

---

## Numerical Methods

The second-order ODE system is solved using Nyström-adapted Runge–Kutta methods, where only the stage derivatives κ̈ (denoted `kdash` in the code) need to be computed — the intermediate κ values are handled analytically. All six methods are specified by their Butcher tableau (α, γ, β) and a flag `plicit` that controls which solver branch is used.

### Explicit Methods (`plicit = 0`)

The stage values are computed directly in a forward sweep. All three explicit methods share the same loop structure; only the tableau differs.

| Method | `method` | Typical m needed |
|---|---|---|
| Explicit Euler | 1 | 30 000 – 50 000 |
| Modified Euler | 2 | 2 200 – 2 600 |
| Heun | 3 | 2 200 – 2 600 |
| Classical RK4 | 4 | 421 – 430 |

The stiffness of the problem forces the explicit methods to use very small step sizes to remain stable, making them expensive. Explicit Euler in particular requires an order of magnitude more steps than RK4.

### Implicit Method — Fixed Point Iteration (`plicit = 1`)

The Hammer–Hollingsworth method (an implicit 4th-order Gauss–Legendre scheme) is solved iteratively: starting from `kdash = 0`, each iteration recomputes all stage values using the previous iterate. The loop runs for `f` iterations per time step (recommended: f = 50). This converges provided the step size is not too large.

| Method | `method` | Typical m needed |
|---|---|---|
| Hammer–Hollingsworth (FP) | 5 | 356 – 360 |

### Implicit Method — Newton Iteration (`plicit = 2`)

The same Hammer–Hollingsworth tableau is used, but the nonlinear stage equations are solved with a matrix Newton iteration. Treating the stage array as a matrix-valued unknown, each step solves:

```
kdash_new = kdash - (I - DF)⁻¹ (kdash - F)
```

where DF is the 4-index Jacobian tensor of the stage equations with respect to `kdash`, assembled from `compute_Jtheta` and `compute_Jkappa` via the chain rule. `np.linalg.tensorsolve` is used to solve the resulting tensor system. Newton iteration converges much faster, allowing the beam motion to be resolved accurately in as few as 10 time steps (recommended: f = 10).

| Method | `method` | Typical m needed |
|---|---|---|
| Hammer–Hollingsworth (Newton) | 6 | 10 – 50 |

---

## Output

Results are displayed as a 3D plot with axes for time t, and the x and y coordinates of the beam. Up to 30 snapshots of the beam shape are overlaid, giving a visual record of how the beam deflects under the applied force and then relaxes. The plot title reports which method was used and how many time steps were taken.

---

## Usage

Open `main.py` and set `demo = 1` to run the full set of examples that replicates the cases shown in the lectures, cycling through all six methods at several step sizes. Set `demo = 0` to use custom inputs:

| Parameter | Variable | Description |
|---|---|---|
| Spatial points | `S` | Number of beam segments (tested with S = 8) |
| Time horizon | `T` | Length of simulation (tested with T = 5) |
| Beam length | `L` | Physical length of the beam |
| Time steps | `m` | See table above for recommended values per method |
| Method | `method` | Integer 1–6, or 0 for a custom Butcher tableau |
| Iterations | `f` | Number of fixed-point or Newton iterations (implicit only) |

To use a custom Runge–Kutta method, set `method = 0` and fill in the `alpha`, `gamma`, `Beta`, and `plicit` fields in the setup block inside `stiffbeam()`.

Then run:

```bash
python main.py
```

---

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```
