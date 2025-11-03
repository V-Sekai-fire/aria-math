# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Common do
  @moduledoc """
  Common types, utilities, and interfaces shared across Axon layers.

  This module provides standardized types and utility functions used by
  all planning and solving components in the axon_layers.
  """

  # ============================================================================
  # Common Type Definitions
  # ============================================================================

  @type goal :: {atom(), any()}
  @type goals :: [goal]
  @type domain :: map()
  @type state :: map()
  @type variable :: atom()
  @type value :: any()
  @type assignment :: %{variable() => value()}
  @type constraint :: (assignment() -> boolean())
  @type constraints :: [constraint]

  @type solution :: %{
          assignment: assignment(),
          stats: %{
            parallel_groups: non_neg_integer(),
            total_goals: non_neg_integer(),
            solve_time_ms: non_neg_integer()
          }
        }

  @type solve_result :: {:ok, solution()} | {:error, String.t()}
  @type consistency_result :: {:consistent, constraints()} | {:inconsistent, String.t()}

  # ============================================================================
  # Common Utility Functions
  # ============================================================================

  @doc """
  Check if a single goal is achieved in the given state.

  This is the standard implementation that works with entity schemas or simple
  state maps (post E-A-V elimination). Multiple modules have similar but stubbed
  versions of this function.
  """
  @spec goal_achieved?(goal(), state()) :: boolean()
  def goal_achieved?({predicate, object}, state) do
    # The subject is implied in the state - check if the predicate has the expected object
    # First try entity-based checking (if state has entities)
    case Map.get(state, predicate) do
      nil -> false
      actual_object -> actual_object == object
    end
  end

  @doc """
  Merge multiple solution assignments.

  This function appears in multiple modules with identical implementations
  that can be shared.
  """
  @spec merge_solutions([solution()]) :: solution()
  def merge_solutions(solutions) do
    # Merge assignments from multiple solutions
    merged_assignment =
      Enum.reduce(solutions, %{}, fn %{assignment: assignment}, acc ->
        Map.merge(acc, assignment)
      end)

    # Combine stats if they exist
    merged_stats = merge_solution_stats(solutions)

    %{assignment: merged_assignment, stats: merged_stats}
  end

  @doc """
  Extract variables from goals.

  Common implementation used across different solvers for variable extraction.
  """
  @spec extract_variables_from_goals(goals()) :: [variable()]
  def extract_variables_from_goals(goals) do
    # Simplified variable extraction for {predicate, value} format
    # Real implementation would analyze domain to find all relevant variables
    goals
    |> Enum.flat_map(fn {predicate, _value} -> [predicate] end)
    |> Enum.uniq()
  end

  @doc """
  Check if domain is consistent with goals.

  Common domain validation function.
  """
  @spec domain_consistent?(goals(), domain()) :: boolean()
  def domain_consistent?(goals, domain) do
    # Check that all predicates in goals are defined in domain
    domain_predicates = Map.get(domain, :predicates, [])

    Enum.all?(goals, fn {predicate, _value} ->
      predicate in domain_predicates
    end)
  end

  @doc """
  Validate that a solution achieves the given goals.

  Common solution validation utility.
  """
  @spec validate_solution(goals(), domain(), solution()) :: boolean()
  def validate_solution(goals, _domain, %{assignment: assignment} = _solution) do
    # Check that all goals are achieved in the solution state
    # This would verify the solution against entity schemas
    Enum.all?(goals, fn goal ->
      # Goal format: {predicate, expected_value} (subject implied by context)
      case goal do
        {predicate, expected_value} ->
          case Map.get(assignment, predicate) do
            nil -> false
            actual_value -> actual_value == expected_value
          end
      end
    end)
  end

  # ============================================================================
  # Common Behaviors/Interfaces
  # ============================================================================

  # Note: Nested behaviors with type references to outer module types
  # can cause compilation issues. These have been removed to prevent
  # cross-referencing problems. Interfaces can be implemented directly
  # in consuming modules for better modularity.

  # ============================================================================
  # Common Helper Functions
  # ============================================================================

  @doc """
  Initialize default solution stats structure.
  """
  @spec default_stats() :: map()
  def default_stats do
    %{
      parallel_groups: 0,
      total_goals: 0,
      solve_time_ms: 0
    }
  end

  @doc """
  Merge stats from multiple solutions.

  Common implementation used by merge_solutions and other functions.
  """
  @spec merge_solution_stats([solution()]) :: map()
  def merge_solution_stats(solutions) do
    defaults = %{goals_solved: 0, elapsed_time: 0}

    Enum.reduce(solutions, defaults, fn %{stats: stats}, acc ->
      Map.update(acc, :goals_solved, 0, &(&1 + Map.get(stats, :goals_solved, 0)))
      Map.update(acc, :elapsed_time, 0, &max(&1, Map.get(stats, :elapsed_time, 0)))
    end)
  end

  @doc """
  Create a standardized error response.
  """
  @spec error_response(String.t()) :: {:error, String.t()}
  def error_response(message) do
    {:error, message}
  end

  @doc """
  Create a standardized success response.
  """
  @spec success_response(any()) :: {:ok, any()}
  def success_response(result) do
    {:ok, result}
  end
end
