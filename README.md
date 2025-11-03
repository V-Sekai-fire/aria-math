# AriaMath

**AI-Accelerated Planning and Tensor Operations**

AriaMath provides a comprehensive suite of planning algorithms and tensor operations for building intelligent systems that can reason about complex scenarios.

## Quick Start

```elixir
# Basic goal solving
alias AriaMath.AxonLayers.GoalSolver
goals = [{:task_done, true}]
{:ok, solution} = GoalSolver.solve_goals(domain, state, goals)

# Temporal reasoning
alias AriaMath.AxonLayers.StnSolver
constraints = [{:start, :end, 5, 10}]
{:ok, schedule} = StnSolver.assign_times(constraints)
```

## Architecture

AriaMath is organized into complementary layers:

### Axon Layers (`axon_layers/`)

GPU-accelerated neural network operations for complex optimization.

### Planning Layers (`planning_layers/`)

Traditional constraint solving and planning algorithms.

## Features

- **Symbolic Planning**: Rule-based goal achievement and temporal reasoning
- **Neural Optimization**: Differentiable models for complex planning problems
- **Unified Data Model**: Consistent AEV (Attribute-Entity-Value) representation
- **GPU Acceleration**: Hardware-accelerated tensor operations via Axon

## Integration

AriaMath powers the planning systems in:

- `aria_planner`: High-level temporal planning
- `aria_forge`: Simulation and scenario generation
- `aria_joint`: Interactive planning environments

## Documentation

See the [full API documentation](https://hexdocs.pm/aria_math/) for detailed usage instructions and examples.
