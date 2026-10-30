
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# -----------------------------------------------------
# PARAMETERS
# -----------------------------------------------------
alpha = 1 / 4
delta_max = alpha / (6 * (1 + alpha))

delta_values = [0.004, 0.011, 0.019, 0.027, 0.032]

# Number of continuous intervals [2k, 2k+1] to plot
N_blocks = 7

colors = plt.cm.viridis(
    np.linspace(0.08, 0.90, len(delta_values))
)

# -----------------------------------------------------
# TIME-SCALE EXPONENTIAL e_alpha(t, 2)
# T = union of [2k, 2k+1], k >= 0
# -----------------------------------------------------
def exp_alpha(t):
    k = int(np.floor(t / 2))
    return (
        ((1 + alpha) * np.exp(alpha)) ** (k - 1)
        * np.exp(alpha * (t - 2 * k))
    )


# -----------------------------------------------------
# HISTORY: x(t) = t + sin(x(t))/2
# Valid for t in [0,1] union {2}
# -----------------------------------------------------
def history(t):
    t = float(t)
    x = t

    for _ in range(80):
        x_new = t + 0.5 * np.sin(x)

        if abs(x_new - x) < 1e-13:
            return x_new

        x = x_new

    return x


# -----------------------------------------------------
# NUMERICAL SOLUTION ON THE HYBRID TIME SCALE
# -----------------------------------------------------
def solve_dynamic_equation(delta):

    segments = {}
    x_start = history(2)

    for k in range(1, N_blocks + 1):

        left = 2 * k
        right = left + 1

        # Delay x(t-2)
        def delayed_value(t):
            if k == 1:
                return history(t - 2)
            else:
                return float(segments[k - 1](t - 2)[0])

        # On right-dense intervals, x^sigma(t) = x(t).
        def rhs(t, y):
            x = y[0]
            x_delay = delayed_value(t)

            return [
                (delta / exp_alpha(t))
                * np.arctan(2 * x + x_delay)
            ]

        sol = solve_ivp(
            rhs,
            (left, right),
            [x_start],
            method="DOP853",
            dense_output=True,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.02
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        segments[k] = sol.sol

        # At t = 2k+1, mu(t)=1, sigma(t)=t+1.
        # Therefore:
        # x(sigma(t)) = x(t) +
        # delta/e_alpha(t,2) *
        # arctan(x(t)+x(sigma(t))+x(t-2)).
        if k < N_blocks:

            x_right = float(sol.sol(right)[0])
            x_delay = delayed_value(right)

            coefficient = delta / exp_alpha(right)

            def jump_equation(z):
                return (
                    z - x_right
                    - coefficient
                    * np.arctan(x_right + z + x_delay)
                )

            radius = coefficient * np.pi / 2 + 1e-9

            x_start = brentq(
                jump_equation,
                x_right - radius,
                x_right + radius
            )

    return segments


# =====================================================
# FIGURE 1: STATE VARIABLE FOR DIFFERENT DELTA
# =====================================================
fig, ax = plt.subplots(figsize=(10, 6))

for delta, color in zip(delta_values, colors):

    segments = solve_dynamic_equation(delta)

    for k in range(1, N_blocks + 1):

        left = 2 * k
        right = left + 1

        t_values = np.linspace(left, right, 120)
        x_values = segments[k](t_values)[0]

        ax.plot(
            t_values,
            x_values,
            color=color,
            linewidth=2.2,
            label=fr"$\delta={delta:.3f}$" if k == 1 else None
        )

        # Show the actual time-scale endpoints.
        ax.scatter(
            [left, right],
            [x_values[0], x_values[-1]],
            color=color,
            s=15,
            zorder=3
        )

ax.set_xlabel("Time $t$", fontsize=12)
ax.set_ylabel("State variable $x(t)$", fontsize=12)

ax.set_title(
    "State Variable for Different Values of $\\delta$\n"
    "$\\alpha=1/4$",
    fontsize=14
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.25, linestyle="--")
ax.set_xlim(2, 2 * N_blocks + 1)

plt.tight_layout()
plt.savefig("state_variable_HURS.png", dpi=300)
plt.show()


# =====================================================
# FIGURE 2: HURS STABILITY CONSTANT VERSUS DELTA
# =====================================================

# nu = (1+alpha)/alpha
nu = (1 + alpha) / alpha

def K2(delta):
    return 0.5 + 3 * nu * delta

def lambda_R(delta):
    return nu / (1 - K2(delta))


delta_range = np.linspace(
    0,
    0.97 * delta_max,
    600
)

lambda_values = lambda_R(delta_range)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    delta_range,
    lambda_values,
    color="#3B4CC0",
    linewidth=2.5,
    label=r"HURS constant $\lambda_R$"
)

# Highlight the chosen delta values
for delta, color in zip(delta_values, colors):

    ax.scatter(
        delta,
        lambda_R(delta),
        color=color,
        s=65,
        zorder=4
    )

# Critical contraction threshold
ax.axvline(
    delta_max,
    color="#D62728",
    linestyle="--",
    linewidth=1.7,
    label=r"Threshold $\delta=1/30$"
)

ax.set_xlabel(r"Parameter $\delta$", fontsize=12)
ax.set_ylabel(r"HURS constant $\lambda_R$", fontsize=12)

ax.set_title(
    "HURS Stability Constant Near the Contraction Threshold\n"
    r"$\alpha=1/4$",
    fontsize=14
)

ax.set_xlim(0, delta_max * 1.02)
ax.grid(True, alpha=0.25, linestyle="--")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("HURS_stability_constant.png", dpi=300)
plt.show()


# -----------------------------------------------------
# PRINT THE VALUES USED IN THE SECOND GRAPH
# -----------------------------------------------------
print("delta        K2          lambda_R")
print("----------------------------------")

for delta in delta_values:
    print(
        f"{delta:.4f}       "
        f"{K2(delta):.4f}       "
        f"{lambda_R(delta):.4f}"
    )
