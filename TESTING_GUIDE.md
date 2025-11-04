# Testing Guide for Ackley Benchmark Example

This guide explains how to test the Ackley benchmark example to ensure it's runnable.

## Prerequisites

First, ensure DeepHyper is installed:

```bash
# Option 1: Install in development mode (recommended for testing local changes)
pip install -e .

# Option 2: Install from PyPI (if you want the official release)
pip install deephyper
```

## Testing Methods

### Method 1: Quick Test Script (Recommended)

We've provided a quick test script that validates all components with minimal evaluations:

```bash
python test_benchmark.py
```

**What it tests:**
- ✓ All required imports
- ✓ Ackley function correctness
- ✓ Problem definition
- ✓ Run function
- ✓ Quick optimization (5 evaluations)

**Expected output:**
```
======================================================================
Testing Ackley Benchmark Example
======================================================================

[1/5] Testing imports...
✓ All imports successful

[2/5] Testing Ackley function...
✓ Ackley at origin: 0.0000000000 (correct)
✓ Ackley at [1,1,1,1,1]: 3.625385

[3/5] Testing problem definition...
✓ Problem created with 5 hyperparameters

[4/5] Defining run function...
✓ Run function defined successfully

[5/5] Running quick optimization test (5 evaluations)...
✓ Optimization completed: 5 evaluations
✓ Best objective found: -XX.XXXXXX
✓ Best configuration: [...]

======================================================================
✓ All tests passed! The benchmark is ready to run.
======================================================================
```

**Time:** ~5-10 seconds

---

### Method 2: Syntax Validation

Check if the Python syntax is correct without running:

```bash
python -m py_compile examples/examples_bbo/plot_benchmark_ackley_optimization.py
```

**Expected output:** (no output means success)

---

### Method 3: Import Test

Verify all imports work:

```bash
python -c "
import sys
sys.path.insert(0, 'src')

from deephyper.hpo import HpProblem, CBO, RandomSearch
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.analysis.hpo import parameters_at_max

print('✓ All imports successful')
"
```

---

### Method 4: Quick Partial Run

Run just the visualization section (no optimization):

```bash
python -c "
import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Define Ackley function
def ackley_function(x, a=20, b=0.2, c=2 * np.pi):
    x = np.array(x)
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    return sum_sq_term + cos_term + a + np.exp(1)

# Test visualization code
x_range = np.linspace(-5, 5, 50)
y_range = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = ackley_function([X[i, j], Y[i, j], 0, 0, 0])

fig, ax = plt.subplots()
ax.contourf(X, Y, Z, levels=20)
plt.savefig('ackley_test.png')
print('✓ Visualization code works! Saved to ackley_test.png')
"
```

---

### Method 5: Full Benchmark Run

Run the complete benchmark with full evaluations (takes longer):

```bash
python examples/examples_bbo/plot_benchmark_ackley_optimization.py
```

**What it does:**
- Runs CBO with 100 evaluations
- Runs Random Search with 100 evaluations
- Generates all visualizations
- Compares algorithms
- Saves results to `results.csv`

**Time:** ~1-3 minutes (depending on your system)

**Expected behavior:**
- Progress bars for both CBO and Random Search
- Multiple matplotlib plots displayed
- Final summary showing CBO outperforms Random Search
- Results saved to disk

---

### Method 6: Headless Mode (for servers without display)

If running on a server without a display:

```bash
# Set matplotlib to non-interactive backend
export MPLBACKEND=Agg
python examples/examples_bbo/plot_benchmark_ackley_optimization.py
```

Or modify the script to save plots instead of showing them:

```python
# Add at the top after imports
import matplotlib
matplotlib.use('Agg')

# Replace plt.show() with plt.savefig()
# plt.show()  # Comment this out
plt.savefig('convergence_plot.png')  # Add this
```

---

### Method 7: Test with Different Settings

Create a quick test with minimal evaluations:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
import numpy as np
from deephyper.hpo import HpProblem, CBO
from deephyper.evaluator import Evaluator

# Define Ackley
def ackley_function(x, a=20, b=0.2, c=2 * np.pi):
    x = np.array(x)
    d = len(x)
    return (-a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
            - np.exp(np.sum(np.cos(c * x) / d)) + a + np.exp(1))

def run(job):
    x = [job.parameters[f'x{i}'] for i in range(5)]
    return -ackley_function(x)

# Quick test: 2D problem, 10 evaluations
problem = HpProblem()
problem.add_hyperparameter((-32.768, 32.768), 'x0')
problem.add_hyperparameter((-32.768, 32.768), 'x1')

evaluator = Evaluator.create(run, method='thread', method_kwargs={'num_workers': 2})
search = CBO(problem, random_state=42, verbose=1)
results = search.search(evaluator, max_evals=10)

print(f'\n✓ Quick test completed: {len(results)} evaluations')
print(f'Best objective: {results[\"objective\"].max():.6f}')
"
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'ConfigSpace'`

**Solution:** Install DeepHyper with all dependencies:
```bash
pip install -e .
```

### Issue: `TypeError: Job.__init__() missing required arguments`

**Solution:** Don't create Job objects manually. Let the Evaluator create them.

### Issue: `ValueError: run_function is not a coroutine`

**Solution:** Use `method="thread"` instead of `method="serial"` for regular (non-async) functions.

### Issue: Plots don't display

**Solution:** Either:
1. Run in an environment with display (X11, Jupyter, etc.)
2. Use non-interactive backend: `export MPLBACKEND=Agg`
3. Modify to save plots: Replace `plt.show()` with `plt.savefig('plot.png')`

### Issue: `KeyError` when using `parameters_at_max`

**Solution:** The API has changed. Use:
```python
# Correct (current API)
best_config, best_objective = parameters_at_max(results)

# Incorrect (old API)
# i_max, best_config = parameters_at_max(results, 1)
```

---

## Verification Checklist

Use this checklist to confirm everything works:

- [ ] Python syntax check passes
- [ ] All imports work
- [ ] Ackley function returns 0 at origin
- [ ] Problem definition creates 5 hyperparameters
- [ ] Quick optimization completes successfully
- [ ] Visualization code runs (if applicable)
- [ ] Full benchmark completes (optional)
- [ ] Results are saved to CSV
- [ ] CBO outperforms Random Search

---

## Performance Benchmarks

Expected performance on a typical system:

| Test Type | Evaluations | Workers | Time |
|-----------|-------------|---------|------|
| Quick test | 5 | 1 | ~5 sec |
| Small test | 20 | 2 | ~15 sec |
| Full benchmark | 100 | 4 | ~1-2 min |
| Large benchmark | 500 | 8 | ~5-10 min |

---

## Next Steps

Once all tests pass:

1. **Run full benchmark:** `python examples/examples_bbo/plot_benchmark_ackley_optimization.py`
2. **Examine results:** Check `results.csv` and generated plots
3. **Modify parameters:** Try different dimensions, evaluation budgets, or algorithms
4. **Create your own:** Use this as a template for other benchmark functions

---

## Additional Resources

- **DeepHyper Documentation:** https://deephyper.readthedocs.io
- **Example Gallery:** `/home/user/deephyper/examples/`
- **Ackley Function Reference:** Wikipedia or optimization literature
- **Report Issues:** https://github.com/deephyper/deephyper/issues
