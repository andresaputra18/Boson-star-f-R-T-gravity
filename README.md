# Boson Star BVP Solver (GR + $f(R,T)$ Gravity)

A Python solver for **static, spherically symmetric boson stars**, formulated as a nonlinear **boundary-value problem (BVP)** and solved with SciPy’s `scipy.integrate.solve_bvp`.

You choose a **central scalar amplitude** `phi0`, and the solver returns self-consistent radial profiles for:

- scalar field amplitude **$\phi(r)$** and **$\psi(r) = \frac{d\phi}{dr}$**
- metric functions **$A(r)$** and **$B(r)$** (spherical metric ansatz)
- the eigenfrequency **$\omega$** (found as part of the BVP)

## What the numerics are doing

- Unknown vector:
  $y(r) = [\phi(r),\ \psi(r),\ A(r),\ B(r)]$
- This is an eigenvalue-type BVP: **4 ODEs** plus **1 free parameter**.
- Internally the solver uses a parameter **$\sigma$** and maps it smoothly to **$\omega$** so that $$0 < \omega < \sqrt{\eta m^2}$$ throughout iterations (helps stability and avoids hard clamping).
- Outer boundary uses a Robin condition consistent with a Yukawa tail.

---

## Requirements

- Python **3.9+** (3.10+ recommended)
- `numpy`
- `scipy`

Optional (plotting):
- `matplotlib`

Install:
```bash
pip install numpy scipy matplotlib
```

---

## Repository layout

Minimal layout:

```text
.
├── bs_solver.py
├── README.md
└── LICENSE
```

---

## How to run

```python
import bs_solver as solv

sol = solv.bs(
    phi0=0.2712,
    m2=1.0,
    lam=0,
    zeta=0,
    R=500.0,
    N=5000,
    tol=1e-7,
)

print(sol["success"], sol["message"])
if sol["success"]:
    print("omega =", sol["omega"])
    print("M     =", sol["M"])
    print("Q     =", sol["Q"])
    print("R_eff =", sol["R_eff"])
    print("C_eff =", sol["C_eff"])
```

---

## Plotting example

```python
import bs_solver as solv
import matplotlib.pyplot as plt

sol = solv.bs(phi0=0.2712, m2=1.0, lam=0, zeta=0, R=500.0, N=5000, tol=1e-7)
assert sol["success"], sol["message"]

r   = sol["r"]
phi = sol["phi"]
m   = sol["m_of_r"]

plt.figure()
plt.plot(r, phi)
plt.xlabel("r")
plt.ylabel(r"$\phi(r)$")
plt.title("Scalar profile")
plt.show()

plt.figure()
plt.plot(r, m)
plt.xlabel("r")
plt.ylabel("m(r)")
plt.title("Enclosed mass profile")
plt.show()
```

---

## API

### `bs(phi0, m2=1.0, lam=0.0, zeta=0.0, R=..., N=..., tol=..., ...)`

**Most-used numerical knobs:**
- `phi0`: central scalar amplitude (main family parameter)
- `R`: outer radius
- `N`: mesh size for postprocessing grid
- `tol`: tolerance passed to `solve_bvp`

**Physics knobs:**
- `m2`: mass term parameter $m^2$
- `lam`: self-interaction strength $\lambda$
- `zeta`: $f(R,T)$ parameter $\zeta$

**Warm-start knobs (useful for parameter scans):**
- `y_guess`: tuple `(r, phi, psi, A, B)` from a previous successful run
- `sigma_guess`: previous solution’s internal `p_sigma` value

---

## Outputs (dict)

If `sol["success"] == True`, the dictionary commonly contains:

| Key | Meaning |
|---|---|
| `success` | solver status |
| `message` | solver message from SciPy |
| `r` | radial grid used for returned arrays |
| `phi`, `psi` | scalar field and derivative |
| `A`, `B` | metric functions |
| `omega` | eigenfrequency $\omega$ |
| `mu` | tail scale $\mu=\sqrt{\eta m^2-\omega^2}$ |
| `m_of_r` | Misner-Sharp mass profile used in the code |
| `M` | total mass |
| `Q` | Noether charge |
| `R_eff` | radius where $m(R_\mathrm{eff}) = f M$ (default $f=0.99$ to get $R_{99}$) |
| `C_eff` | effective compactness $M/R_\mathrm{eff}$ |
| `p_sigma` | internal σ parameter used to map to $\omega$ |
| `phi0, m2, lam, zeta, R` | echoed inputs |

If `success` is `False`, you’ll only get:
- `success`
- `message`

---

## Practical scan tips
When scanning `phi0` (let's say to make $M-\omega$ plot), reuse the last solution as a guess (warm start):

```python
import bs_solver as solv

phis = [0.10, 0.15, 0.20, 0.25]
prev_guess = None
prev_sigma = None

for phi0 in phis:
    sol = solv.bs(phi0=phi0, y_guess=prev_guess, sigma_guess=prev_sigma, R=400.0, N=4000, tol=1e-6)
    print(phi0, sol["success"], sol.get("omega"), sol.get("M"))
    if sol["success"]:
        prev_guess = (sol["r"], sol["phi"], sol["psi"], sol["A"], sol["B"])
        prev_sigma = sol["p_sigma"]
```

**Tip:** For large parameter scans, it’s convenient to write a “sweeper” code that:
- steps through `phi0` (or `zeta`, `lam`, …),
- warm-starts from the previous solution (`y_guess`, `sigma_guess`),
- records a compact table of derived quantities (e.g., `omega`, `M`, `Q`, `R_eff`, `C_eff`),
- and periodically saves checkpoints so long runs can resume.

---

## License

This project is licensed under the **GNU General Public License v3.0**.

---

## Citation

If you use this solver in academic work, please cite the associated paper describing the model and equations.

> NOTE: The manuscript is currently under review. An arXiv link and the final publisher DOI/link will be added here once available.

## Paper data (CSV)

All datasets used to generate the plots in the associated paper are included in this repository under `data/`. Each CSV is named by figure/panel (e.g., `fig01.csv`).
