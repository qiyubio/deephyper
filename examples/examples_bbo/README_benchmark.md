# Ackley Function Benchmark Example

This benchmark example demonstrates DeepHyper's optimization capabilities on the Ackley function, a classic multimodal test function.

## Overview

**File**: `plot_benchmark_ackley_optimization.py`

The Ackley function is a widely used benchmark in optimization research due to:
- Large, nearly flat outer region
- Numerous local minima
- Single global minimum at the origin f(0,...,0) = 0

## Features Demonstrated

1. **Black-box function optimization**: 5-dimensional Ackley function
2. **Multiple search algorithms**: CBO (Centralized Bayesian Optimization) vs Random Search
3. **Parallel evaluation**: Using thread-based evaluator with 4 workers
4. **Comprehensive visualization**:
   - Function landscape visualization
   - Convergence trajectory comparison
   - Parameter evolution over time
5. **Performance analysis**: Quantitative comparison of algorithms

## Running the Benchmark

```bash
python examples/examples_bbo/plot_benchmark_ackley_optimization.py
```

## Expected Results

- **CBO** should significantly outperform random search
- Convergence to near-optimal solution (close to zero) within 100 evaluations
- Clear visualization of the optimization landscape and search progress

## Requirements

The example requires the standard DeepHyper installation:

```bash
pip install deephyper
```

No additional dependencies are needed beyond the base installation.

## Benchmark Metrics

The example reports:
- Best objective value found
- Actual Ackley function value at the best point
- Distance from the global minimum
- Percentage improvement of CBO over random search

## Extending the Benchmark

You can easily modify this benchmark to:
- Test different dimensionalities (change the number of x parameters)
- Use different search algorithms (RegularizedEvolution, etc.)
- Adjust the evaluation budget (change `max_evals`)
- Compare different surrogate models or acquisition functions
- Add more workers for larger-scale parallelism

## References

- Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing". Springer.
- DeepHyper documentation: https://deephyper.readthedocs.io
