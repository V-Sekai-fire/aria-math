# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.GoalSolver.Utils do
  @moduledoc """
  Utility functions for GoalSolver.

  This module provides specific utility functions and delegates common
  operations to the shared AriaMath.AxonLayers.Common module.
  """

  # Import common types and functions
  alias AriaMath.AxonLayers.Common

  # Re-export common types for convenience
  @type goal :: Common.goal()
  @type goals :: Common.goals()
  @type domain :: Common.domain()
  @type solution :: Common.solution()

  @doc """
  Check if a single goal is achieved in the given state.

  Delegated to Common.goal_achieved?/2 for consistent implementation.
  """
  @spec goal_achieved?(goal(), map()) :: boolean()
  defdelegate goal_achieved?(goal, state), to: Common

  @doc """
  Validate that a solution achieves the given goals.

  Delegated to Common.validate_solution/3 for consistent implementation.
  """
  @spec validate_solution(goals(), domain(), solution()) :: boolean()
  defdelegate validate_solution(goals, domain, solution), to: Common

  @doc """
  Merge multiple solution assignments.

  Delegated to Common.merge_solutions/1 for consistent implementation.
  """
  @spec merge_solutions([solution()]) :: solution()
  defdelegate merge_solutions(solutions), to: Common

  @doc """
  Check if domain is consistent with goals.

  Delegated to Common.domain_consistent?/2 for consistent implementation.
  """
  @spec domain_consistent?(goals(), domain()) :: boolean()
  defdelegate domain_consistent?(goals, domain), to: Common

  @doc """
  Extract variables from goals.

  Delegated to Common.extract_variables_from_goals/1 for consistent implementation.
  """
  @spec extract_variables_from_goals(goals()) :: Common.variable()
  defdelegate extract_variables_from_goals(goals), to: Common

  # ============================================================================
  # GoalSolver-Specific Utilities
  # ============================================================================

  @doc """
  Goal achievement helper that applies goal correction logic specific to GoalSolver.

  This wraps the common goal_achieved?/2 but may add GoalSolver-specific logic.
  """
  @spec goal_achieved_goal_solver?(goal(), map()) :: boolean()
  def goal_achieved_goal_solver?(goal, state) do
    # For now, delegate directly to common implementation
    # In the future, this could add GoalSolver-specific corrections
    Common.goal_achieved?(goal, state)
  end

  @doc """
  Specialized solution validation with GoalSolver-specific checks.
  """
  @spec validate_goal_solver_solution(goals(), domain(), solution()) :: boolean()
  def validate_goal_solver_solution(goals, domain, solution) do
    # Use common validation
    Common.validate_solution(goals, domain, solution)
    # Could add GoalSolver-specific validation here
  end
end
