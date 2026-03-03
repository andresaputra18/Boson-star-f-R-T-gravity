# Boson Star BVP Solver (GR + $f(R,T)$ Gravity)

A Python solver for **static, spherically symmetric boson stars**, formulated as a nonlinear **boundary-value problem (BVP)** and solved with SciPy’s `scipy.integrate.solve_bvp`.

You choose a **central scalar amplitude** `phi0`, and the solver returns self-consistent radial profiles for:

- scalar field amplitude **$\tilde{\phi}(\tilde{r})$** and **$\tilde{\psi}(\tilde{r}) = \frac{d\tilde{\phi}}{d\tilde{r}}$**
- metric functions **$A(\tilde{r})$** and **$B(\tilde{r})$** (spherical metric ansatz)
- the eigenfrequency **$\tilde{\omega}$** (found as part of the BVP)

## What the numerics are doing

- Unknown vector:
  $y(\tilde{r}) = [\tilde{\phi}(\tilde{r}),\ \tilde{\psi}(\tilde{r}),\ A(\tilde{r}),\ B(\tilde{r})]$
- This is an eigenvalue-type BVP: **4 ODEs** plus **1 free parameter**.
- Internally the solver uses a parameter **$\sigma$** and maps it smoothly to **$\tilde{\omega}$** so that $$0 < \tilde{\omega} < \sqrt{\eta}$$ throughout iterations (helps stability and avoids hard clamping).
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
- `m2`: mass term parameter $m^2$ (default is 1 since other quantities are rescaled to $m$)
- `lam`: self-interaction strength $\tilde{\lambda}$
- `zeta`: rescaled $f(R,T)$ parameter $\zeta = \frac{\alpha}{\kappa}$

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
| `mu` | tail scale $\mu=\sqrt{\eta - \tilde{\omega}^2}$ |
| `m_of_r` | Misner-Sharp mass profile $\tilde{M}_{\rm MS}$ used in the code |
| `M` | total mass $\tilde{M}$ |
| `Q` | Noether charge |
| `R_eff` | radius where enclosed Misner–Sharp mass at R_eff equals $f\tilde{M}$ (default f = 0.99 → $\tilde{R}_{99}$) |
| `C_eff` | effective compactness $\tilde{M}/\tilde{R}_{\rm eff}$ |
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

---

## Paper data (CSV)

All datasets used to generate the plots in the associated paper are stored in `data/` as CSV files.  
Each file is named using the pattern:

`FIG[number]_[type]_zeta_[zetanumber]_lambda_[lambdanumber].csv`

### Fields

- **`number`**: figure/plot index from **1** to **18** (e.g., `FIG1`, `FIG18`).
- **`type`**: dataset category
  - **`bs`** — parameter sweep outputs: $M$, $Q$, $C$, $\phi_0$, $\omega$, $R_{99}$
  - **`tov`** — TOV force analysis outputs
  - **`ec`** — energy condition tests (margins listed below)

### Energy condition margins (`ec`)

When `type = ec`, the CSV includes the following energy-condition margin columns:

- **`ec1`**: $\rho_{\mathrm{eff}}$
- **`ec2`**: $\rho_{\mathrm{eff}} - p_{r,\mathrm{eff}}$
- **`ec3`**: $\rho_{\mathrm{eff}} - p_{t,\mathrm{eff}}$
- **`ec4`**: $\rho_{\mathrm{eff}} + p_{r,\mathrm{eff}}$
- **`ec5`**: $\rho_{\mathrm{eff}} + p_{t,\mathrm{eff}}$
- **`ec6`**: $\rho_{\mathrm{eff}} + p_{r,\mathrm{eff}} + 2p_{t,\mathrm{eff}}$

### Encoding of $\zeta$

`zetanumber` is a **4-digit code** representing $\zeta$:

- `0200` $\rightarrow \zeta = 0.200$
- for negative values, prefix with `m`:
  - `m0200` $\rightarrow \zeta = -0.200$

### Encoding of $\tilde{\lambda}$

`lambdanumber` is a **7-digit code** representing $\tilde{\lambda}$, with the numeric value given by:

$$
\tilde{\lambda} = \frac{\texttt{lambdanumber}}{10^5}.
$$

Examples:

- `1000000` $\rightarrow \tilde{\lambda} = 10.000$
- `0687500` $\rightarrow \tilde{\lambda} = 6.875$
- for negative values, prefix with `m`:
  - `m0687500` $\rightarrow \tilde{\lambda} = -6.875$

### Examples

- `FIG3_bs_zeta_0200_lambda_0687500.csv`  
  $\rightarrow$ `FIG3`, `bs`, $\zeta = 0.200$, $\tilde{\lambda} = 6.875$

- `FIG12_ec_zeta_m0200_lambda_1000000.csv`  
  $\rightarrow$ `FIG12`, `ec`, $\zeta = -0.200$, $\tilde{\lambda} = 10.000$
