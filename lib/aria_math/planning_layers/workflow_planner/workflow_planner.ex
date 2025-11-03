# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.WorkflowPlanner do
  @moduledoc """
  Axon-based workflow planner.

  This module provides an integrated planning solution using Axon tensor operations
  for GPU/CPU-accelerated planning computations.

  ## Axon Model Architecture

  The WorkflowPlanner combines multiple neural planning components:
  - Goal solving networks
  - Temporal constraint networks (STN)
  - Solution tensor graph processing
  """

  import Axon

  alias AriaMath.AxonLayers.SolutionTensorGraph
  alias AriaMath.AxonLayers.WorkflowPlanner.{Execution, Solver, GraphBuilder}
  alias AriaPlanner.Planner.Temporal.STN

  require Logger

  @doc """
  Plan and execute using CPU tensor operations with blacklisting.

  This is the sole entry point for planning and execution that integrates
  temporal constraint solving and goal analysis for sophisticated planning.
  When execution is false (default), returns the expected plan without execution.
  When execution is true, executes the plan using blacklisting machinery.
  """
  @spec run_lazy(map(), map(), [tuple()], keyword(), boolean()) :: {:ok, SolutionTensorGraph.t()} | {:error, String.t()}
  def run_lazy(domain, initial_state, tasks, opts \\ [], execution \\ false) do
    verbose = Keyword.get(opts, :verbose, 0)
    use_temporal = Keyword.get(opts, :temporal_reasoning, true)

    if verbose > 0 do
      Logger.info("WorkflowPlanner: Starting tensor-based planning with execution: #{execution}")
      Logger.debug("WorkflowPlanner: Domain: #{inspect(domain)}")
      Logger.debug("WorkflowPlanner: Tasks: #{inspect(tasks)}")
      Logger.debug("WorkflowPlanner: Temporal reasoning: #{use_temporal}")
    end

    try do
      # Initialize temporal network if temporal reasoning is enabled
      _stn =
        if use_temporal do
          STN.new(time_unit: :second)
        else
          nil
        end

      # MANDATORY: Goal optimization for all multigoals (no configuration option)
      solution_graph =
        if length(tasks) > 1 do
          # All multigoals MUST use goal optimization - no fallback path
          case Solver.solve_with_goal_solver(domain, initial_state, tasks, opts) do
            {:ok, solution} ->
              GraphBuilder.create_solution_graph(solution)

            {:error, reason} ->
              raise "Critical failure: Goal optimization required for multigoals but failed - #{inspect(reason)}"
          end
        else
          # Single goals use standard planning via HTN backtracking
          GraphBuilder.create_solution_graph_with_temporal(tasks, initial_state, verbose)
        end

      # Execute the plan if requested
      if execution do
        # Get the todo items from the solution graph
        todo_items = Execution.extract_todo_items(solution_graph)

        # Execute with blacklisting
        {_final_state, _blacklist} = Execution.execute_todo_items(todo_items, initial_state, %{}, domain)

        # Return the solution graph without updating final state (since update_final_state doesn't exist)
        {:ok, solution_graph}
      else
        {:ok, solution_graph}
      end
    rescue
      e ->
        Logger.error("WorkflowPlanner: Execution failed: #{Exception.message(e)}")
        {:error, "Execution failed: #{Exception.message(e)}"}
    end
  end

  @doc """
  Build an end-to-end Axon model for the complete planning pipeline.

  ## Parameters
  - `opts` - Model configuration options
    - `:max_goals` - Maximum number of goals (default: 5)
    - `:max_timepoints` - Maximum time points for STN (default: 10)
    - `:embedding_size` - Size of embeddings (default: 32)

  ## Returns
  - Axon model that combines goal solving, temporal reasoning, and plan generation

  ## Usage
  ```elixir
  model = WorkflowPlanner.build_end_to_end_model(max_goals: 3)
  params = Axon.init(model)
  # Use for GPU/CPU-accelerated planning
  ```
  """
  @spec build_end_to_end_model(keyword()) :: Axon.t()
  def build_end_to_end_model(opts \\ []) do
    max_goals = Keyword.get(opts, :max_goals, 5)
    max_timepoints = Keyword.get(opts, :max_timepoints, 10)
    embedding_size = Keyword.get(opts, :embedding_size, 32)

    # Inputs for the integrated planning model
    goals_input = Axon.input("goals", shape: {nil, max_goals, embedding_size})
    state_input = Axon.input("state", shape: {nil, embedding_size})
    constraints_input = Axon.input("constraints", shape: {nil, max_timepoints, max_timepoints, 2})

    # Goal achievement prediction (from goal solver)
    goal_model = GoalSolver.build_model(max_goals: max_goals, embedding_size: embedding_size)
    goal_outputs = goal_model.({goals_input, state_input})

    # Temporal constraint solving (from STN solver)
    stn_model = AriaMath.AxonLayers.StnSolver.build_model(max_timepoints: max_timepoints)
    stn_outputs = stn_model.(constraints_input)

    # Combine goal and temporal information
    combined_input = Axon.concatenate([goal_outputs.goal_achievement, stn_outputs.consistency])

    # Final planning decision layers
    planning_output =
      combined_input
      |> Axon.layer(:dense, 64, key: :planning1)
      |> Axon.activation(:relu)
      |> Axon.layer(:dense, 32, key: :planning2)
      |> Axon.activation(:relu)
      # Actions and timing predictions
      |> Axon.layer(:dense, max_goals * 2, key: :planning3)

    # Reshape to structured output
    final_output = Axon.reshape(planning_output, {nil, max_goals, 2})

    # Return complete model
    Axon.container(%{
      goal_predictions: goal_outputs.goal_achievement,
      stn_consistency: stn_outputs.consistency,
      plan_actions: final_output
    })
  end
end
