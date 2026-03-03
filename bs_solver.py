# -----------------------------------------------------------------------------
# Boson star BVP solver (GR + optional f(R,T) tweak)
#
# What this file does (in plain words):
#   Given a central scalar amplitude `phi0`, solve for a static, spherically
#   symmetric, ground-state boson star by turning the field equations into a
#   nonlinear boundary-value problem and letting SciPy's `solve_bvp` do the work.
#
# Model snapshot (kept deliberately brief):
#   Metric    : ds² = −A(r)² dt² + B(r)² dr² + r² dΩ²
#   Scalar    : Φ(r, t) = φ(r) e^(i ω t)
#   Potential : U(φ) = m² φ² + (λ/2) φ⁴   (m² = m2, λ = lam)
#   Unknowns  : y = [φ, ψ, A, B] with ψ = dφ/dr
#
# f(R,T) knob:
#   The parameter `zeta` (ζ) modifies the equations through
#     η = (1 + 4ζ)/(1 + 2ζ).
#   Setting ζ = 0 recovers the GR expressions used here.
#
# Numerics snapshot:
#   - Solve a 4-ODE + 1-parameter BVP (σ is the free parameter).
#   - Reparameterize the eigenfrequency as ω(σ) to keep
#       0 < ω < √(η m²)
#     smoothly, without hard clamps during Newton iterations.
#   - Impose a Robin outer boundary condition based on the Yukawa tail
#     scale μ = √(η m² − ω²).
#   - Use a μ-aware mesh to place extra points where the solution needs them.
# -----------------------------------------------------------------------------
import numpy as np
from scipy.integrate import solve_bvp

import warnings
# Silence harmless RuntimeWarnings from SciPy's BVP internals (division-by-zero guards, etc.).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.integrate._bvp")

def bs(
    phi0,
    m2=1.0,
    lam=0.0,
    r0=1e-6,
    R=140.0,
    N=1100,
    sigma0=2.0,
    tol=1e-6,
    y_guess=None,
    sigma_guess=None,
    guardrails=True,
    clip_AB=(1e-6, 5.0),
    clip_field=5.0,
    zeta=0.0,
    mass_fraction=0.99,
):
    """
    Solve a spherically symmetric, ground-state boson star as a nonlinear BVP.
    
    Provide the central scalar amplitude `phi0`. The solver then finds the
    profiles φ(r), ψ(r)=dφ/dr, and the metric functions A(r), B(r), together with
    the eigenfrequency ω. Setting `zeta=0` gives the GR version used here.
    
    Numerical notes
    ---------------
    - The free parameter is σ, mapped to ω(σ) so ω always stays in (0, √(η m²)).
    - The outer boundary uses a Robin condition consistent with φ ~ exp(−μ r)/r,
      where μ = √(η m² − ω²).
    - A μ-aware mesh is used so the tail is resolved even when μ is small.
    
    Parameters
    ----------
    phi0 : float
        Central scalar amplitude φ(0). This is the main "dial" you turn to scan a
        family of solutions.
    m2, lam : float
        Potential parameters in U(φ) = m² φ² + (λ/2) φ⁴.
    r0 : float
        Small inner radius used in place of r=0 to avoid explicit 1/r terms.
    R : float
        Outer radius. If ω is close to √(η m²), the decay length 1/μ grows, so R
        should be large enough that μ·R is comfortably bigger than O(10).
    N : int
        Number of mesh points for the initial collocation grid (the solver may
        refine this internally).
    sigma0 : float
        Initial σ used to build a first guess (ignored if `sigma_guess` is set).
    tol : float
        Nonlinear tolerance passed to `scipy.integrate.solve_bvp`.
    y_guess : tuple or None
        Optional warm start as (r, φ, ψ, A, B). Values are interpolated onto the
        new mesh.
    sigma_guess : float or None
        Optional warm start for σ.
    guardrails : bool
        If True, clip trial Newton iterates to avoid temporary blow-ups. This does
        not affect the converged solution; it just keeps the solver sane.
    clip_AB : (float, float)
        Min/max clamps for A and B when `guardrails=True`.
    clip_field : float
        Absolute clamp for |φ| and |ψ| when `guardrails=True`.
    zeta : float
        f(R,T) parameter ζ.
    mass_fraction : float
        Used to define R_eff such that m(R_eff) = mass_fraction · M.
    
    Returns
    -------
    dict
        If the solve succeeds, the dict contains:
          - success, message
          - omega, mu, p_sigma
          - r, phi, psi, A, B
          - m_of_r, M
          - Q
          - R_eff, C_eff
          - plus echoes of (R, phi0, m2, lam, zeta)
    
        If the solve fails, only `success` and `message` are returned.
    """
    
    # f(R,T) bookkeeping: η rescales the effective mass term.
    eta = (1.0 + 4.0*zeta) / (1.0 + 2.0*zeta)
    eta = max(eta, 1e-12)
    m = np.sqrt(m2)

    # Potential U(φ) and its derivative with respect to φ² (as used in the reduced equations).
    def U(phi):         return m2*phi**2 + (lam/2.0)*phi**4
    def dU_dphi2(phi):  return m2 + lam*phi**2

    # Eigenfrequency map.
    def omega(sig):
        wcap = np.sqrt(eta*m2)
        return 0.5*wcap*(1.0 + 0.999999999999*np.tanh(sig))

    # Build a μ-aware radial mesh: more points near the center and in the far tail when μ is small.
    def make_mesh(r0, R, N, w_now):
        mu0 = float(np.sqrt(max(0.0, eta*m2 - w_now*w_now)))
        mu_eff = max(mu0, 1e-6)
        # outer clustering grows as μ→0
        t = max(0.0, min(1.0, (0.20 - mu_eff)/0.20))
        a = 2.2
        b = 2.0 + 6.0*t
        s = np.linspace(0.0, 1.0, int(N))
        g = (s**a) / (s**a + (1.0 - s)**b)
        r = r0 + (R - r0)*g

        # enforce strict monotonicity & exact endpoints
        r[0], r[-1] = r0, R
        dr = np.diff(r); eps = 1e-12
        dr[dr < eps] = eps
        r = r0 + np.concatenate(([0.0], np.cumsum(dr)))
        r *= (R - r0) / (r[-1] - r0); r += (r0 - r[0]); r[-1] = R
        return r

    # Right-hand side of the ODE system for y = [φ, ψ, A, B].
    def rhs(r, y, p):
        phi, psi, A, B = y
        w = omega(p[0])

        if guardrails:
            A_eval   = np.clip(A, clip_AB[0], clip_AB[1])
            B_eval   = np.clip(B, clip_AB[0], clip_AB[1])
            phi_eval = np.clip(phi, -clip_field, clip_field)
            psi_eval = np.clip(psi, -clip_field, clip_field)
        else:
            A_eval, B_eval, phi_eval, psi_eval = A, B, phi, psi

        re = np.maximum(r, r0)  # pad all 1/r terms

        # GR pieces (ζ = 0), written in a solver-friendly form.
        A_base = (A_eval*B_eval**2)/(2.0*re) - A_eval/(2.0*re) \
               + (B_eval**2)*re*(w**2)*phi_eval**2/(2.0*A_eval) \
               - (A_eval*B_eval**2)*re*U(phi_eval)/2.0 + (A_eval*re*psi_eval**2)/2.0

        B_base =  B_eval/(2.0*re) - (B_eval**3)/(2.0*re) \
               + (B_eval**3)*re*(w**2)*phi_eval**2/(2.0*A_eval**2) \
               + (B_eval**3)*re*U(phi_eval)/2.0 + (B_eval*re*psi_eval**2)/2.0

        # f(R,T) corrections: added linearly in ζ on top of the GR pieces.
        B_corr = (B_eval**3)*re*(w**2)*phi_eval**2/(A_eval**2) \
               + 2.0*(B_eval**3)*re*U(phi_eval) \
               + (B_eval*re)*psi_eval**2

        A_corr = (B_eval**2)*re*(w**2)*phi_eval**2/(A_eval) \
               - 2.0*A_eval*(B_eval**2)*re*U(phi_eval) \
               + A_eval*re*psi_eval**2

        A_p = A_base + zeta*A_corr
        B_p = B_base + zeta*B_corr

        psi_p =  (B_eval**2)*phi_eval*dU_dphi2(phi_eval)*eta \
               - (B_eval**2)*(w**2)*phi_eval/(A_eval**2) \
               + (B_p/B_eval - A_p/A_eval - 2.0/re)*psi_eval

        phi_p = psi_eval if guardrails else psi
        return np.vstack([phi_p, psi_p, A_p, B_p])

    # Boundary conditions: regular center + asymptotic (Robin) decay at r=R.
    def bc(ya, yb, p):
        phi_a, psi_a, A_a, B_a = ya
        phi_b, psi_b, A_b, B_b = yb
        w   = omega(p[0])
        mu2 = eta*m2 - w*w
        mu  = np.sqrt(mu2) if (mu2 > 0.0) else 0.0
        return np.array([
            phi_a - phi0,                # center: φ(r0) = φ0 (shooting parameter / family label)
            psi_a - 0.0,                 # center: regularity requires ψ(r0) = 0
            B_a   - 1.0,                 # center: fix radial gauge with B(r0) = 1
            A_b   - 1.0,                 # outer: normalize time coordinate with A(R) = 1
            psi_b + (mu + 1.0/R)*phi_b   # outer: Robin decay matching φ ∝ e^{−μ r}/r
        ])

    # Initial guess: small-r series smoothly blended into an exponential tail.
    def series_coeffs(w):
        U0 = U(phi0); D0 = dU_dphi2(phi0)
        phi2 = (phi0/6.0)*(eta*D0 - w*w)
        B2   = (1.0/6.0)*(w*w*phi0**2 + U0) + (zeta/3.0)*(w*w*phi0**2 + 2.0*U0)
        A2   = (1.0/6.0)*(2.0*w*w*phi0**2 - U0) - (zeta/3.0)*(w*w*phi0**2 - 4.0*U0)
        return A2, B2, phi2

    def seed_with_series_and_tail(r, w, r_match=2.0, width=1.0):
        A2, B2, phi2 = series_coeffs(w)
        phi_s = phi0 + phi2*r**2
        psi_s = 2.0*phi2*r
        A_s   = 1.0 + A2*r**2
        B_s   = 1.0 + B2*r**2

        mu_g  = max(1e-8, np.sqrt(max(0.0, eta*m2 - w*w)))
        rr    = np.maximum(r, 1e-9)
        phi_f = (phi0*np.exp(-mu_g*np.clip(rr - r_match, 0.0, None))) / (1.0 + rr)
        psi_f = -(mu_g + 1.0/(1.0 + rr)) * phi_f
        A_f   = np.ones_like(r); B_f = np.ones_like(r)

        s = 0.5*(1.0 - np.tanh((r - r_match)/width))
        phi0r = s*phi_s + (1.0 - s)*phi_f
        psi0r = s*psi_s + (1.0 - s)*psi_f
        A0r   = s*A_s   + (1.0 - s)*A_f
        B0r   = s*B_s   + (1.0 - s)*B_f
        return np.vstack([phi0r, psi0r, A0r, B0r])

    # Assemble the mesh and a starting guess for the Newton iterations.
    w0 = omega(sigma0 if sigma_guess is None else sigma_guess)
    r  = make_mesh(r0, R, N, w0)
    if y_guess is None:
        y0 = seed_with_series_and_tail(r, w0)
    else:
        rg, phig, psig, Ag, Bg = y_guess
        y0 = np.vstack([
            np.interp(r, rg, phig),
            np.interp(r, rg, psig),
            np.interp(r, rg, Ag),
            np.interp(r, rg, Bg),
        ])
    p0 = np.array([sigma0 if sigma_guess is None else sigma_guess], dtype=float)

    # Hand it to SciPy's collocation solver.
    sol = solve_bvp(rhs, bc, r, y0, p=p0, tol=tol, max_nodes=200000)
    out = {'success': bool(sol.success), 'message': sol.message}
    if not sol.success:
        return out

    # Post-processing on a uniform grid: compute profiles + derived quantities.
    rr = np.linspace(r0, R, int(N))
    phi, psi, A, B = sol.sol(rr)
    w  = float(omega(sol.p[0]))
    mu = float(np.sqrt(max(0.0, eta*m2 - w*w)))

    # Misner–Sharp mass profile m(r).
    m_of_r = 0.5*rr*(1.0 - 1.0/(B**2))
    m_mon  = np.maximum.accumulate(m_of_r)
    M      = float(m_mon[-1])

    # Noether charge (including the (1 + 2ζ) prefactor used in this setup).
    integrand = (B/A) * (rr**2) * (phi**2)
    Q = float(w*(1.0 + 2.0*zeta)*np.trapz(integrand, rr))

    # Effective radius: where the enclosed mass reaches a chosen fraction of the total, and the corresponding compactness.
    def mass_fraction_radius(r, m_of_r, frac=0.99):
        # Walk the (monotone) mass profile and interpolate the radius where m(r)=frac·M.
        # Handle non-monotone numerics by taking a cumulative maximum.
        r = np.asarray(r, float)
        m = np.asarray(m_of_r, float)
        if r.size == 0 or m.size == 0 or r.size != m.size:
            return float('nan')
            
        if not (0.0 < frac <= 1.0): 
            raise ValueError("frac must be in (0,1].")
            
        m_mon = np.maximum.accumulate(m); Mtot = m_mon[-1]
        if not np.isfinite(Mtot) or Mtot <= 0.0: 
            return float('nan')
        target = frac*Mtot
        
        if m_mon[-1] < target: 
            return float(r[-1])
        if not np.all(np.diff(r) >= 0):
            idx = np.argsort(r)
            r = r[idx]
            m_mon = m_mon[idx]
            
        idx = np.searchsorted(m_mon, target, side='left')
        if idx == 0:
            return float(r[0])
            
        r0i, r1i = r[idx-1], r[idx]
        m0, m1 = m_mon[idx-1], m_mon[idx]
        if m1 == m0 or r1i == r0i: 
            return float(r1i)
            
        return float(r0i + (r1i - r0i) * (target - m0) / (m1 - m0))

    R_eff = mass_fraction_radius(rr, m_of_r, frac=mass_fraction)
    C_eff = M / R_eff if R_eff > 0 else float('inf')

    out.update({
        'omega': w,
        'mu': mu,
        'r': rr,
        'phi': phi,
        'psi': psi,
        'A': A,
        'B': B,
        'm_of_r': m_of_r,
        'M': M,
        'Q': Q,
        'p_sigma': float(sol.p[0]),
        'R': float(rr[-1]),
        'phi0': float(phi0),
        'R_eff': R_eff,
        'C_eff': C_eff,
        'm2': float(m2),
        'lam': float(lam),
        'zeta': float(zeta)
    })
    return out