# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.WorkflowPlanner.GraphBuilder do
  @moduledoc """
  Graph building functions for WorkflowPlanner.

  This module contains functions for creating solution tensor graphs
  from planning solutions and temporal information.
  """

  alias AriaMath.AxonLayers.SolutionTensorGraph

  require Logger

  @doc """
  Create a solution graph with temporal information.
  """
  @spec create_solution_graph_with_temporal([tuple()], map(), non_neg_integer()) ::
          {:ok, SolutionTensorGraph.t()}
  def create_solution_graph_with_temporal(tasks, _initial_state, verbose) do
    # Create solution tree with temporal information
    solution_tree = %{
      root: %{
        type: :compound,
        subtasks:
          Enum.map(tasks, fn {task_name, _args} ->
            {:primitive, task_name}
          end),
        temporal_constraints: %{
          start_time: 0,
          # Estimate duration
          duration: length(tasks) * 5,
          end_time: length(tasks) * 5
        }
      }
    }

    solution_tensor_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

    if verbose > 0 do
      Logger.info("WorkflowPlanner: Planning completed with temporal reasoning")
      Logger.debug("WorkflowPlanner: Solution graph has #{solution_tensor_graph.num_nodes} nodes")
    end

    {:ok, solution_tensor_graph}
  end

  @doc """
  Create a solution graph from a solver result.
  """
  @spec create_solution_graph(map()) :: {:ok, SolutionTensorGraph.t()}
  def create_solution_graph(solution) do
    # Create solution tree from goal solver result
    solution_tree = %{
      root: %{
        type: :compound,
        subtasks: [:goal_solved],
        solution: solution
      }
    }

    solution_tensor_graph = SolutionTensorGraph.from_solution_tree(solution_tree)
    {:ok, solution_tensor_graph}
  end
end
