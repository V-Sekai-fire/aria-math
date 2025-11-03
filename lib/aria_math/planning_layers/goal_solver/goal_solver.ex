# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.GoalSolver do
  @moduledoc """
  Axon-based goal solver.

  This module implements differentiable goal solving using neural network operations
  for GPU/CPU-accelerated planning computations.

  ## Axon Model Architecture

  The goal solver uses an Axon model that processes goals and state through:
  - Goal encoding layers
  - Constraint satisfaction networks
  - Solution prediction layers

  ## Submodules

  - `AriaMath.AxonLayers.GoalSolver.Model` - Neural network model building
  - `AriaMath.AxonLayers.GoalSolver.Core` - Core solving algorithms
  - `AriaMath.AxonLayers.GoalSolver.Utils` - Utility functions
  """

  alias AriaMath.AxonLayers.GoalSolver.{Model, Core, Utils}
  alias AriaMath.AxonLayers.Common
  alias AriaPlanner.Planner.PlannerMetadata

  # Use common types
  @type goal :: Common.goal()
  @type goals :: Common.goals()
  @type domain :: Common.domain()
  @type state :: Common.state()
  @type assignment :: Common.assignment()
  @type solution :: Common.solution()
  @type solve_result :: Common.solve_result()

  @doc """
  Solve goals using dependency analysis and parallel solving.

  This is the main entry point that matches the MiniZinc interface.
  """
  @spec solve_goals(domain(), state(), goals(), keyword()) :: solve_result()
  def solve_goals(domain, state, goals, options \\ []) do
    # When calling into the parallel solver, we need to pass planner metadata.
    # This assumes planner_metadata is available in the `options` keyword list.
    planner_metadata = Keyword.get(options, :planner_metadata)
    Core.solve_goals_parallel(domain, state, goals, options, planner_metadata)
  end

  @doc """
  Build an Axon model for goal solving.

  Delegated to AriaMath.AxonLayers.GoalSolver.Model.build_model/1
  """
  @spec build_model(keyword()) :: any()
  defdelegate build_model(opts \\ []), to: Model

  @doc """
  Check if domain is consistent with goals.

  Delegated to AriaMath.AxonLayers.GoalSolver.Utils.domain_consistent?/2
  """
  @spec domain_consistent?(goals(), domain()) :: boolean()
  defdelegate domain_consistent?(goals, domain), to: Utils

  @doc """
  Check if a single goal is achieved in the given state.

  Works with entity schemas or simple state maps (post E-A-V elimination).

  Delegated to AriaMath.AxonLayers.GoalSolver.Utils.goal_achieved?/2
  """
  @spec goal_achieved?(goal(), state()) :: boolean()
  defdelegate goal_achieved?(goal, state), to: Utils

  @doc """
  Validate that a solution achieves the given goals.

  Delegated to AriaMath.AxonLayers.GoalSolver.Utils.validate_solution/3
  """
  @spec validate_solution(goals(), domain(), solution()) :: boolean()
  defdelegate validate_solution(goals, domain, solution), to: Utils

  @doc """
  Merge multiple solution assignments.

  Delegated to AriaMath.AxonLayers.GoalSolver.Utils.merge_solutions/1
  """
  @spec merge_solutions([solution()]) :: solution()
  defdelegate merge_solutions(solutions), to: Utils

  @doc """
  Solve a component of goals (public interface).

  Delegated to AriaMath.AxonLayers.GoalSolver.Core.solve_component/5
  """
  @spec solve_component(goals(), domain(), state(), keyword(), PlannerMetadata.t() | nil) ::
          {:ok, solution()} | {:error, String.t()}
  defdelegate solve_component(goals, domain, state, options \\ [], planner_metadata \\ nil), to: Core

  @doc """
  Extract variables from goals (alias for extract_variables_from_goals).

  Delegated to AriaMath.AxonLayers.GoalSolver.Core.extract_variables/1
  """
  @spec extract_variables(goals()) :: [atom()]
  defdelegate extract_variables(goals), to: Core
end
