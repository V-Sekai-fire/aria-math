# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath do
  @moduledoc """
  AriaMath: AI-Accelerated Planning and Tensor Operations

  AriaMath provides a comprehensive suite of planning algorithms and tensor operations
  for building intelligent systems that can reason about complex scenarios.

  ## Architecture Overview

  The library is organized into two complementary layers that work together:

  ### Axon Layers (`axon_layers/`)
  **Technology**: Neural Networks & GPU Acceleration

  Specialized components built with Elixir's Axon framework for machine learning
  and differentiable computing. These handle neural network-based components:

  - `GoalSolver.Model` - Neural goal optimization models
  - `StnSolver.Axon` - Differentiable STN constraint solving

  ### Planning Layers (`planning_layers/`)
  **Technology**: Classical Algorithms & Traditional Computing

  Traditional planning and constraint solving algorithms, focusing on symbolic
  reasoning and graph-based approaches:

  - `Common` - Shared types, utilities, and AEV data model
  - `SolutionTensorGraph` - Graph representations for planning solutions
  - `GoalSolver` - Symbolic goal optimization with constraint satisfaction
  - `StnSolver` - Floyd-Warshall algorithm for temporal constraint networks
  - `WorkflowPlanner` - Plan execution and blacklisting machinery

  ## Data Model: AEV (Attribute-Entity-Value)

  All planning components use a consistent **AEV semantic data model**:

  - **Attribute/Predicate**: What property (e.g., `:task_done`, `:robot_at`)
  - **Entity/Subject**: Which object (often implicit in goals/context)
  - **Value**: Property value (e.g., `true`, `charging_station`)

  This provides a unified approach to goal representation, state management,
  and constraint evaluation across all planning components.

  ## Integration

  The layers integrate seamlessly:
  - Planning layers handle symbolic reasoning and constraint propagation
  - Axon layers provide neural acceleration for complex optimization
  - Common shared utilities ensure consistent behavior

  ## Usage

  ### Basic Goal Solving
  ```elixir
  alias AriaMath.AxonLayers.GoalSolver

  goals = [{:task_done, true}]
  domain = %{...}  # Planning domain
  state = %{...}   # Current state

  {:ok, solution} = GoalSolver.solve_goals(domain, state, goals)
  ```

  ### Temporal Reasoning
  ```elixir
  alias AriaMath.AxonLayers.StnSolver

  constraints = [{:start, :end, 5, 10}]
  {:ok, schedule} = StnSolver.assign_times(constraints)
  ```

  ### Neural Goal Optimization
  ```elixir
  alias AriaMath.AxonLayers.GoalSolver.Model

  model = Model.build_model(max_goals: 10, embedding_size: 64)
  ```

  ## Getting Started

  1. **Planning Basics**: Start with `GoalSolver` for basic goal achievement
  2. **Temporal Planning**: Use `StnSolver` for time-constrained planning
  3. **Neural Acceleration**: Integrate `GoalSolver.Model` for learned optimization
  4. **Advanced Workflows**: Leverage `WorkflowPlanner` for complex plan execution

  ## Performance Considerations

  - **Planning Layers**: Optimized for CPU-based symbolic computation
  - **Axon Layers**: GPU-accelerated for large-scale optimization problems
  - **Hybrid Approach**: Combine both layers for best performance

  ## Related Applications

  AriaMath powers the planning systems in:

  - **aria_planner**: High-level temporal planning and goal optimization
  - **aria_forge**: Simulation and scenario generation
  - **aria_joint**: Interactive planning environments
  """

  @type goal :: AriaMath.AxonLayers.Common.goal()
  @type goals :: AriaMath.AxonLayers.Common.goals()
  @type domain :: AriaMath.AxonLayers.Common.domain()
  @type state :: AriaMath.AxonLayers.Common.state()
end
