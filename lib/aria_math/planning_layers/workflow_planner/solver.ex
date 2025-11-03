# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.WorkflowPlanner.Solver do
  @moduledoc """
  Solver functions for WorkflowPlanner.

  This module contains functions for goal solving and constraint creation.
  """

  @doc """
  Solve goals using the goal solver.
  """
  @spec solve_with_goal_solver(map(), map(), [tuple()], keyword()) ::
          {:ok, map()} | {:error, String.t()}
  def solve_with_goal_solver(domain, initial_state, tasks, opts) do
    alias AriaMath.AxonLayers.GoalSolver

    # Convert tasks to goals format expected by GoalSolver
    goals =
      Enum.map(tasks, fn {task_name, args} ->
        {task_name, :execute, args}
      end)

    # Create constraints based on domain knowledge
    _constraints = create_goal_constraints(domain, tasks)

    case GoalSolver.solve_goals(domain, initial_state, goals, opts) do
      {:ok, solution} ->
        # Convert goal solver solution back to our format
        final_state = solution.assignment[:final_state] || initial_state
        {:ok, %{state: final_state, plan: solution.assignment}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Create goal constraints based on domain knowledge.
  """
  @spec create_goal_constraints(map(), [tuple()]) :: [function()]
  def create_goal_constraints(domain, tasks) do
    # Create basic constraints based on domain knowledge
    # This is a simplified implementation - in practice, this would be more sophisticated
    Enum.map(tasks, fn {task_name, _args} ->
      fn _assignment ->
        # Basic constraint: task must be executable in domain
        Map.has_key?(domain.actions, task_name) or Map.has_key?(domain.methods, task_name)
      end
    end)
  end
end
