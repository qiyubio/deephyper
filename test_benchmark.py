#!/usr/bin/env python
"""
Quick test script for the Ackley benchmark example.
This runs a minimal version to verify all imports and core functionality work.
"""

import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("Testing Ackley Benchmark Example")
print("=" * 70)

# Test 1: Check all imports
print("\n[1/5] Testing imports...")
try:
    import numpy as np
    from deephyper.hpo import HpProblem, CBO, RandomSearch
    from deephyper.evaluator import Evaluator
    from deephyper.evaluator.callback import TqdmCallback
    from deephyper.analysis.hpo import parameters_at_max
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test Ackley function
print("\n[2/5] Testing Ackley function...")
def ackley_function(x, a=20, b=0.2, c=2 * np.pi):
    x = np.array(x)
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    return sum_sq_term + cos_term + a + np.exp(1)

# Test at origin (should be ~0)
result = ackley_function([0, 0, 0, 0, 0])
assert abs(result) < 1e-10, f"Expected ~0, got {result}"
print(f"✓ Ackley at origin: {result:.10f} (correct)")

# Test at a known point
result2 = ackley_function([1, 1, 1, 1, 1])
print(f"✓ Ackley at [1,1,1,1,1]: {result2:.6f}")

# Test 3: Test problem definition
print("\n[3/5] Testing problem definition...")
problem = HpProblem()
for i in range(5):
    problem.add_hyperparameter((-32.768, 32.768), f"x{i}")
print(f"✓ Problem created with {len(problem)} hyperparameters")

# Test 4: Define run function
print("\n[4/5] Defining run function...")
def run(job):
    config = job.parameters
    x = [config[f"x{i}"] for i in range(5)]
    value = ackley_function(x)
    return -value  # Negate for maximization

print(f"✓ Run function defined successfully")

# Test 5: Run a quick optimization (serial, minimal evals)
print("\n[5/5] Running quick optimization test (5 evaluations)...")
try:
    evaluator = Evaluator.create(
        run,
        method="thread",  # Use thread evaluator
        method_kwargs={"num_workers": 1},
    )

    search = CBO(problem, random_state=42, verbose=0)
    results = search.search(evaluator, max_evals=5)

    print(f"✓ Optimization completed: {len(results)} evaluations")

    # Check results structure
    assert "objective" in results.columns, "Missing objective column"
    assert "job_id" in results.columns, "Missing job_id column"

    # Get best result
    best_config, best_obj = parameters_at_max(results)
    print(f"✓ Best objective found: {best_obj:.6f}")

    # Show best configuration
    best_x = [best_config[f"x{i}"] for i in range(5)]
    print(f"✓ Best configuration: [{', '.join([f'{x:.4f}' for x in best_x])}]")

except Exception as e:
    print(f"✗ Optimization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All tests passed! The benchmark is ready to run.")
print("=" * 70)
print("\nTo run the full benchmark with 100 evaluations and visualizations:")
print("  python examples/examples_bbo/plot_benchmark_ackley_optimization.py")
