r"""
Benchmark: Ackley Function Optimization
========================================

**Author(s)**: DeepHyper Team

This example demonstrates a complete benchmark using DeepHyper to optimize the
Ackley function, a classic multimodal benchmark function commonly used to test
optimization algorithms. The Ackley function is challenging due to its large
search space and numerous local minima.

The benchmark includes:

* Definition of the Ackley function (5-dimensional)
* Comparison of different search algorithms (CBO vs Random Search)
* Parallel evaluation using multiple workers
* Comprehensive results analysis and visualization

The global minimum of the Ackley function is at f(0,...,0) = 0.
"""

# %%
# Installation
# ------------
#
# First, ensure DeepHyper is installed:
#
# .. code-block:: bash
#
#    pip install deephyper

# %%
# Imports
# -------
#
# .. dropdown:: Import statements

import numpy as np
import matplotlib.pyplot as plt

from deephyper.hpo import HpProblem, CBO, RandomSearch
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.analysis.hpo import (
    parameters_at_max,
    plot_search_trajectory_single_objective_hpo,
)

# %%
# Define the Ackley Function
# ---------------------------
#
# The Ackley function is a widely used benchmark in optimization. It features:
#
# * A large, nearly flat outer region
# * A large number of local minima
# * A single global minimum at the origin
#
# The function is defined as:
#
# .. math::
#
#     f(x) = -a \exp\left(-b\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}\right) -
#     \exp\left(\frac{1}{d}\sum_{i=1}^d \cos(cx_i)\right) + a + \exp(1)
#
# where typically a=20, b=0.2, c=2π, and the search domain is [-32.768, 32.768]^d.


def ackley_function(x, a=20, b=0.2, c=2 * np.pi):
    """Compute the Ackley function value.

    Args:
        x (array-like): Input vector of dimension d.
        a (float): Parameter a (default: 20).
        b (float): Parameter b (default: 0.2).
        c (float): Parameter c (default: 2π).

    Returns:
        float: The Ackley function value (to be minimized).
    """
    x = np.array(x)
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    return sum_sq_term + cos_term + a + np.exp(1)


def run(job):
    """The black-box function to optimize.

    This function extracts parameters from the job configuration,
    evaluates the Ackley function, and returns the negative value
    (since DeepHyper maximizes by default, but we want to find the minimum).

    Args:
        job: A job object containing the hyperparameters to evaluate.

    Returns:
        float: The negative Ackley function value (for maximization).
    """
    config = job.parameters

    # Extract the parameters (5-dimensional optimization problem)
    x = [config[f"x{i}"] for i in range(5)]

    # Evaluate the Ackley function
    # Return negative value because DeepHyper maximizes, but we want to minimize
    value = ackley_function(x)

    return -value  # Negate for maximization


# %%
# Visualize the Ackley Function
# -------------------------------
#
# Let's visualize a 2D slice of the Ackley function to understand its landscape:

# Create a 2D grid for visualization
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = ackley_function([X[i, j], Y[i, j], 0, 0, 0])

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Surface plot (contour)
contour = ax1.contourf(X, Y, Z, levels=20, cmap="viridis")
ax1.contour(X, Y, Z, levels=10, colors="black", alpha=0.3, linewidths=0.5)
ax1.plot(0, 0, "r*", markersize=15, label="Global minimum")
ax1.set_xlabel("x0")
ax1.set_ylabel("x1")
ax1.set_title("Ackley Function (2D slice)")
ax1.legend()
plt.colorbar(contour, ax=ax1, label="f(x)")

# 3D-like representation
ax2.contour(X, Y, Z, levels=20, cmap="viridis")
ax2.plot(0, 0, "r*", markersize=15, label="Global minimum")
ax2.set_xlabel("x0")
ax2.set_ylabel("x1")
ax2.set_title("Ackley Function Contours")
ax2.legend()

plt.tight_layout()
plt.show()

# %%
# Define the Search Space
# ------------------------
#
# We define a 5-dimensional optimization problem where each dimension
# has the typical Ackley function bounds [-32.768, 32.768]:

problem = HpProblem()

# Add 5 hyperparameters, one for each dimension
for i in range(5):
    problem.add_hyperparameter((-32.768, 32.768), f"x{i}")

print(problem)

# %%
# Create the Evaluator
# --------------------
#
# We use a thread-based evaluator with 4 workers for parallel evaluation.
# The TqdmCallback provides a progress bar during the search:

evaluator = Evaluator.create(
    run,
    method="thread",
    method_kwargs={
        "num_workers": 4,
        "callbacks": [TqdmCallback()],
    },
)

print(f"Evaluator created with {evaluator.num_workers} workers")

# %%
# Run Bayesian Optimization with CBO
# ------------------------------------
#
# Now we run the Centralized Bayesian Optimization (CBO) algorithm
# to find the minimum of the Ackley function:

print("\n" + "=" * 60)
print("Running CBO (Centralized Bayesian Optimization)")
print("=" * 60)

search_cbo = CBO(
    problem,
    random_state=42,
    log_dir="cbo_results",
    verbose=0,
)

results_cbo = search_cbo.search(evaluator, max_evals=100)
print(f"\nCBO completed! Total evaluations: {len(results_cbo)}")

# %%
# Run Random Search Baseline
# ---------------------------
#
# For comparison, we also run a random search baseline:

print("\n" + "=" * 60)
print("Running Random Search (Baseline)")
print("=" * 60)

search_random = RandomSearch(
    problem,
    random_state=42,
    log_dir="random_results",
)

results_random = search_random.search(evaluator, max_evals=100)
print(f"\nRandom Search completed! Total evaluations: {len(results_random)}")

# %%
# Analyze CBO Results
# -------------------
#
# Let's examine the best configuration found by CBO:

i_max, best_config_cbo = parameters_at_max(results_cbo, 1)
best_objective_cbo = results_cbo.loc[i_max, "objective"].iloc[0]

print("\n" + "=" * 60)
print("CBO Best Results")
print("=" * 60)
print(f"Best objective value: {best_objective_cbo:.6f}")
print(f"Best configuration:")
for key, value in best_config_cbo.items():
    print(f"  {key}: {value:.6f}")

# Compute the actual Ackley value (remember we negated it)
best_x_cbo = [best_config_cbo[f"x{i}"] for i in range(5)]
actual_ackley_value_cbo = ackley_function(best_x_cbo)
print(f"\nActual Ackley function value: {actual_ackley_value_cbo:.6f}")
print(f"Distance from global minimum (0,0,0,0,0): {np.linalg.norm(best_x_cbo):.6f}")

# %%
# Analyze Random Search Results
# ------------------------------

i_max, best_config_random = parameters_at_max(results_random, 1)
best_objective_random = results_random.loc[i_max, "objective"].iloc[0]

print("\n" + "=" * 60)
print("Random Search Best Results")
print("=" * 60)
print(f"Best objective value: {best_objective_random:.6f}")
print(f"Best configuration:")
for key, value in best_config_random.items():
    print(f"  {key}: {value:.6f}")

best_x_random = [best_config_random[f"x{i}"] for i in range(5)]
actual_ackley_value_random = ackley_function(best_x_random)
print(f"\nActual Ackley function value: {actual_ackley_value_random:.6f}")
print(f"Distance from global minimum: {np.linalg.norm(best_x_random):.6f}")

# %%
# Compare Search Trajectories
# ----------------------------
#
# Visualize the convergence of both algorithms:

fig, ax = plt.subplots(figsize=(10, 6))

# Plot CBO trajectory
plot_search_trajectory_single_objective_hpo(
    results_cbo, show=False, ax=ax, label="CBO"
)

# Plot Random Search trajectory
results_random_sorted = results_random.sort_values("job_id")
cummax = results_random_sorted["objective"].cummax()
ax.plot(
    results_random_sorted["job_id"],
    cummax,
    label="Random Search",
    linewidth=2,
    alpha=0.7,
)

ax.set_xlabel("Number of Evaluations")
ax.set_ylabel("Best Objective Found (negative Ackley)")
ax.set_title("Convergence Comparison: CBO vs Random Search")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Performance Summary
# -------------------
#
# Let's create a summary comparing the two approaches:

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

improvement = (
    (best_objective_cbo - best_objective_random) / abs(best_objective_random) * 100
)

summary_data = {
    "Algorithm": ["CBO", "Random Search"],
    "Best Objective": [best_objective_cbo, best_objective_random],
    "Actual Ackley Value": [actual_ackley_value_cbo, actual_ackley_value_random],
    "Distance from Optimum": [
        np.linalg.norm(best_x_cbo),
        np.linalg.norm(best_x_random),
    ],
}

print(f"\n{'Algorithm':<20} {'Best Objective':<18} {'Ackley Value':<18} {'Distance':<12}")
print("-" * 70)
for i in range(len(summary_data["Algorithm"])):
    print(
        f"{summary_data['Algorithm'][i]:<20} "
        f"{summary_data['Best Objective'][i]:<18.6f} "
        f"{summary_data['Actual Ackley Value'][i]:<18.6f} "
        f"{summary_data['Distance from Optimum'][i]:<12.6f}"
    )

print(f"\nCBO Improvement over Random Search: {improvement:.2f}%")
print(f"Global minimum (target): f(0,0,0,0,0) = 0")

# %%
# Visualize Parameter Distribution
# ----------------------------------
#
# Let's examine how the parameters evolved during the CBO search:

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i in range(5):
    ax = axes[i]
    param_name = f"x{i}"

    # Plot parameter values over time
    ax.scatter(
        results_cbo["job_id"],
        results_cbo[f"p:{param_name}"],
        c=results_cbo["objective"],
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2, label="Global optimum")
    ax.set_xlabel("Evaluation Number")
    ax.set_ylabel(f"{param_name}")
    ax.set_title(f"Parameter {param_name} Evolution")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()

# Remove extra subplot
axes[5].remove()

plt.tight_layout()
plt.show()

# %%
# Conclusion
# ----------
#
# This benchmark demonstrates:
#
# 1. **CBO Effectiveness**: The Centralized Bayesian Optimization algorithm
#    significantly outperforms random search on the Ackley function.
#
# 2. **Convergence**: CBO converges much faster towards the global minimum,
#    making efficient use of the evaluation budget.
#
# 3. **Parameter Space Exploration**: CBO intelligently explores the parameter
#    space, focusing on promising regions around the global minimum.
#
# 4. **Practical Usage**: This example shows how to set up a complete benchmark
#    with DeepHyper, including problem definition, parallel evaluation,
#    multiple algorithm comparison, and comprehensive analysis.
#
# The Ackley function is just one of many benchmark functions. DeepHyper can be
# applied to any black-box optimization problem, including hyperparameter
# optimization for machine learning models, neural architecture search,
# and scientific simulation calibration.
