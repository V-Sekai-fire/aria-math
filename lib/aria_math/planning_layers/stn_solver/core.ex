# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.StnSolver.Core do
  @moduledoc """
  Core STN algorithms and data structures.

  This module contains the traditional algorithms for STN consistency checking,
  constraint propagation, and time assignment using the Floyd-Warshall algorithm.
  """

  require Logger

  @type time_point :: atom()
  @type constraint :: {time_point(), time_point(), number(), number()}
  @type constraints :: [constraint()]
  @type distance_matrix :: %{time_point() => %{time_point() => number()}}
  @type consistency_result :: {:consistent, constraints()} | {:inconsistent, String.t()}

  @doc """
  Check if a Simple Temporal Network is consistent using Floyd-Warshall algorithm.

  ## Parameters
  - `constraints` - List of temporal constraints in format {from, to, min_duration, max_duration}

  ## Returns
  - `{:consistent, tightened_constraints}` - Network is consistent with all implied constraints
  - `{:inconsistent, reason}` - Network contains conflicting constraints

  ## Algorithm Details
  This function uses the Floyd-Warshall algorithm to propagate constraints and detect
  negative cycles that indicate temporal inconsistencies.
  """
  @spec check_consistency(constraints()) :: consistency_result()
  def check_consistency(constraints) do
    # Validate constraint format (let ArgumentError propagate)
    validate_constraints!(constraints)

    try do
      # Extract all time points from constraints
      time_points = extract_time_points(constraints)

      # Build initial distance matrix from constraints
      build_result = build_distance_matrix(constraints, time_points)

      case build_result do
        {:ok, distance_matrix} ->
          # Apply Floyd-Warshall algorithm for constraint propagation
          fw_result = floyd_warshall(distance_matrix, time_points)

          case fw_result do
            {:ok, tightened_matrix} ->
              # Convert back to constraint format
              tightened_constraints = matrix_to_constraints(tightened_matrix, time_points)
              {:consistent, tightened_constraints}

            {:error, reason} ->
              {:inconsistent, reason}
          end

        {:error, reason} ->
          {:inconsistent, reason}
      end
    rescue
      error ->
        Logger.error("STN consistency check failed: #{Exception.message(error)}")
        {:inconsistent, "Internal error during consistency check"}
    end
  end

  @doc """
  Assign concrete time values to time points in a consistent STN.

  ## Parameters
  - `constraints` - List of temporal constraints
  - `reference_time` - Optional reference time point (defaults to first time point)

  ## Returns
  - `{:ok, assignments}` - Time point assignments as %{time_point => time_value}
  - `{:error, reason}` - Failed to assign times (inconsistent network)

  ## Examples

      iex> constraints = [{:start, :end, 5, 10}]
      iex> AriaMath.AxonLayers.StnSolver.Core.assign_times(constraints)
      {:ok, %{start: 0, end: 7.5}}
  """
  @spec assign_times(constraints(), time_point() | nil) ::
          {:ok, %{time_point() => number()}} | {:error, String.t()}
  def assign_times(constraints, reference_time \\ nil) do
    consistency_result = check_consistency(constraints)

    case consistency_result do
      {:consistent, tightened_constraints} ->
        # Use tightened constraints for proper assignment
        time_points = extract_time_points(constraints)

        # Handle empty constraints case
        if Enum.empty?(time_points) do
          # For empty constraints, return default assignment with :start
          {:ok, %{start: 0}}
        else
          # Find disconnected components and assign each independently
          components = find_disconnected_components(tightened_constraints, time_points)

          # Assign times for each component independently
          assignments =
            Enum.reduce(components, %{}, fn component, acc ->
              component_assignments = assign_component_times(tightened_constraints, component, reference_time)
              Map.merge(acc, component_assignments)
            end)

          {:ok, assignments}
        end

      {:inconsistent, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Convert STN struct to internal constraint format.
  """
  @spec stn_to_constraints(any()) :: constraints()
  def stn_to_constraints(stn) do
    # Convert STN constraints to our internal format
    # STN constraints are typically in format {{from, to}, {min, max}}
    # We need to convert to {from, to, min, max}
    for {{from, to}, {min_val, max_val}} <- stn.constraints do
      {from, to, min_val, max_val}
    end
  end

  @doc """
  Convert internal constraints back to STN format.
  """
  @spec constraints_to_stn(any(), constraints()) :: any()
  def constraints_to_stn(stn, constraints) do
    # Convert our internal constraint format back to STN format
    # Convert from {from, to, min, max} to {{from, to}, {min, max}}
    constraint_map =
      for {from, to, min_val, max_val} <- constraints, into: %{} do
        {{from, to}, {min_val, max_val}}
      end

    # Return updated STN with tightened constraints
    %{stn | constraints: constraint_map, consistent: true}
  end

  # Validate constraint format and raise ArgumentError for invalid constraints
  @spec validate_constraints!(constraints()) :: :ok
  defp validate_constraints!(constraints) do
    Enum.each(constraints, fn constraint ->
      case constraint do
        {from, to, min_duration, max_duration} when is_atom(from) and is_atom(to) ->
          unless is_number(min_duration) do
            raise ArgumentError,
                  "Invalid constraint format: min_duration must be a number, got #{inspect(min_duration)}"
          end

          unless is_number(max_duration) do
            raise ArgumentError,
                  "Invalid constraint format: max_duration must be a number, got #{inspect(max_duration)}"
          end

          unless min_duration <= max_duration do
            raise ArgumentError,
                  "Invalid constraint: min_duration (#{min_duration}) must be <= max_duration (#{max_duration})"
          end

        _ ->
          raise ArgumentError,
                "Invalid constraint format: expected {atom(), atom(), number(), number()}, got #{inspect(constraint)}"
      end
    end)

    :ok
  end

  # Extract all unique time points from constraints
  @spec extract_time_points(constraints()) :: [time_point()]
  defp extract_time_points(constraints) do
    constraints
    |> Enum.flat_map(fn {from, to, _min, _max} -> [from, to] end)
    |> Enum.uniq()
    |> Enum.sort()
  end

  # Build initial distance matrix from constraints
  # In STN terms: distance[i][j] represents the minimum time from i to j
  @spec build_distance_matrix(constraints(), [time_point()]) ::
          {:ok, distance_matrix()} | {:error, String.t()}
  defp build_distance_matrix(constraints, time_points) do
    try do
      # Initialize matrix with infinity (no constraint)
      infinity = :infinity
      matrix = initialize_matrix(time_points, infinity)

      # Add direct constraints
      matrix = add_direct_constraints(matrix, constraints, infinity)

      # Add reflexive constraints (distance from point to itself is 0)
      matrix = add_reflexive_constraints(matrix, time_points)

      {:ok, matrix}
    rescue
      error ->
        {:error, "Failed to build distance matrix: #{Exception.message(error)}"}
    end
  end

  # Initialize distance matrix with infinity values
  @spec initialize_matrix([time_point()], number() | :infinity) :: distance_matrix()
  defp initialize_matrix(time_points, infinity) do
    for from <- time_points, into: %{} do
      {from, for(to <- time_points, into: %{}, do: {to, infinity})}
    end
  end

  # Add direct constraints to the matrix
  @spec add_direct_constraints(distance_matrix(), constraints(), number()) :: distance_matrix()
  defp add_direct_constraints(matrix, constraints, infinity) do
    Enum.reduce(constraints, matrix, fn {from, to, min_duration, max_duration}, acc ->
      # For STN: constraint {from, to, min, max} means:
      # to - from >= min_duration  (i.e., from - to <= -min_duration)
      # to - from <= max_duration  (i.e., to - from <= max_duration)

      # Update distance[from][to] with upper bound on (to - from)
      current_from_to = get_distance(acc, from, to, infinity)
      new_from_to = min(current_from_to, max_duration)
      acc = put_distance(acc, from, to, new_from_to)

      # Update distance[to][from] with upper bound on (from - to)
      current_to_from = get_distance(acc, to, from, infinity)
      new_to_from = min(current_to_from, -min_duration)
      put_distance(acc, to, from, new_to_from)
    end)
  end

  # Add reflexive constraints (distance to self is 0)
  @spec add_reflexive_constraints(distance_matrix(), [time_point()]) :: distance_matrix()
  defp add_reflexive_constraints(matrix, time_points) do
    Enum.reduce(time_points, matrix, fn point, acc ->
      put_distance(acc, point, point, 0)
    end)
  end

  # Floyd-Warshall algorithm for all-pairs shortest paths
  # This propagates all implied constraints in the STN
  @spec floyd_warshall(distance_matrix(), [time_point()]) ::
          {:ok, distance_matrix()} | {:error, String.t()}
  defp floyd_warshall(matrix, time_points) do
    try do
      # Floyd-Warshall: for each intermediate point k, for each pair (i,j),
      # check if path i->k->j is shorter than direct i->j
      result_matrix =
        Enum.reduce(time_points, matrix, fn k, current_matrix ->
          Enum.reduce(time_points, current_matrix, fn i, inner_matrix ->
            Enum.reduce(time_points, inner_matrix, fn j, innermost_matrix ->
              # Check for shorter path: i -> k -> j
              ik_distance = get_distance(innermost_matrix, i, k, :infinity)
              kj_distance = get_distance(innermost_matrix, k, j, :infinity)

              if ik_distance != :infinity and kj_distance != :infinity do
                new_distance = ik_distance + kj_distance
                current_distance = get_distance(innermost_matrix, i, j, :infinity)

                if new_distance < current_distance do
                  put_distance(innermost_matrix, i, j, new_distance)
                else
                  innermost_matrix
                end
              else
                innermost_matrix
              end
            end)
          end)
        end)

      # Check for negative cycles (inconsistent constraints)
      case detect_negative_cycles(result_matrix, time_points) do
        :ok ->
          {:ok, result_matrix}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      error ->
        {:error, "Floyd-Warshall algorithm failed: #{Exception.message(error)}"}
    end
  end

  # Detect negative cycles in the distance matrix
  @spec detect_negative_cycles(distance_matrix(), [time_point()]) :: :ok | {:error, String.t()}
  defp detect_negative_cycles(matrix, time_points) do
    # In STN terms, negative cycles indicate inconsistent constraints
    # A negative cycle exists if distance[i][i] < 0 for any time point i
    negative_cycles =
      Enum.filter(time_points, fn point ->
        distance = get_distance(matrix, point, point, 0)
        distance < 0
      end)

    if Enum.empty?(negative_cycles) do
      :ok
    else
      {:error,
       "Temporal network is inconsistent: negative cycle detected involving time points #{inspect(negative_cycles)}"}
    end
  end

  # Convert distance matrix back to constraint format
  @spec matrix_to_constraints(distance_matrix(), [time_point()]) :: constraints()
  defp matrix_to_constraints(matrix, time_points) do
    # Convert tightened distance matrix back to constraint format
    # In STN: distance[from][to] represents the tightest upper bound on (to - from)
    # So constraint {from, to, min, max} where:
    # - max = distance[from][to] (tightest upper bound on to - from)
    # - min = -distance[to][from] (tightest lower bound on to - from, since distance[to][from] is tightest upper bound on from - to)

    for from <- time_points, to <- time_points, from != to do
      distance_from_to = get_distance(matrix, from, to, :infinity)
      distance_to_from = get_distance(matrix, to, from, :infinity)

      if distance_from_to != :infinity and distance_to_from != :infinity do
        # Convert distances back to constraint bounds
        max_val = distance_from_to
        min_val = -distance_to_from

        # Only include if we have meaningful bounds and they're valid
        if is_number(max_val) and is_number(min_val) and min_val <= max_val do
          {from, to, min_val, max_val}
        end
      end
    end
    |> Enum.reject(&is_nil/1)
  end

  # Find minimum constraint between two time points
  @spec find_min_constraint(constraints(), time_point(), time_point()) :: number() | nil
  defp find_min_constraint(constraints, from, to) do
    Enum.find_value(constraints, fn
      {^from, ^to, min_val, _max_val} -> min_val
      # Reverse constraint
      {^to, ^from, min_val, _max_val} -> -min_val
      _ -> nil
    end)
  end

  # Find maximum constraint between two time points
  @spec find_max_constraint(constraints(), time_point(), time_point()) :: number() | nil
  defp find_max_constraint(constraints, from, to) do
    Enum.find_value(constraints, fn
      {^from, ^to, _min_val, max_val} -> max_val
      # Reverse constraint
      {^to, ^from, _min_val, max_val} -> -max_val
      _ -> nil
    end)
  end

  # Find disconnected components in the STN using DFS traversal
  @spec find_disconnected_components(constraints(), [time_point()]) :: [[time_point()]]
  defp find_disconnected_components(constraints, time_points) do
    # Build adjacency list from constraints
    adjacency_list = build_adjacency_list(constraints, time_points)

    # Find connected components using DFS
    {components, _visited} =
      Enum.reduce(time_points, {[], MapSet.new()}, fn time_point, {components, visited} ->
        if MapSet.member?(visited, time_point) do
          {components, visited}
        else
          # Start DFS from this unvisited time point
          {component, new_visited} = dfs_component(time_point, adjacency_list, visited)
          {[component | components], new_visited}
        end
      end)

    # Reverse to maintain original order
    Enum.reverse(components)
  end

  # Build adjacency list from constraints
  @spec build_adjacency_list(constraints(), [time_point()]) :: %{time_point() => [time_point()]}
  defp build_adjacency_list(constraints, time_points) do
    # Start with empty adjacency list
    adjacency_list = Enum.reduce(time_points, %{}, &Map.put(&2, &1, []))

    # Add edges from constraints
    Enum.reduce(constraints, adjacency_list, fn {from, to, _min, _max}, acc ->
      # Add bidirectional edges
      acc
      |> Map.update!(from, &[to | &1])
      |> Map.update!(to, &[from | &1])
    end)
  end

  # DFS to find all time points in a connected component
  @spec dfs_component(time_point(), %{time_point() => [time_point()]}, MapSet.t(time_point())) ::
          {[time_point()], MapSet.t(time_point())}
  defp dfs_component(start_point, adjacency_list, visited) do
    {component, final_visited} = dfs_visit(start_point, adjacency_list, visited, [start_point])
    {Enum.reverse(component), final_visited}
  end

  # Recursive DFS visit
  @spec dfs_visit(time_point(), %{time_point() => [time_point()]}, MapSet.t(time_point()), [time_point()]) ::
          {[time_point()], MapSet.t(time_point())}
  defp dfs_visit(current, adjacency_list, visited, component) do
    new_visited = MapSet.put(visited, current)

    # Visit all unvisited neighbors
    neighbors = Map.get(adjacency_list, current, [])

    Enum.reduce(neighbors, {component, new_visited}, fn neighbor, {comp_acc, visited_acc} ->
      if MapSet.member?(visited_acc, neighbor) do
        {comp_acc, visited_acc}
      else
        dfs_visit(neighbor, adjacency_list, visited_acc, [neighbor | comp_acc])
      end
    end)
  end

  # Assign times for a single connected component
  @spec assign_component_times(constraints(), [time_point()], time_point() | nil) :: %{time_point() => number()}
  defp assign_component_times(constraints, component_time_points, reference_time) do
    # Choose reference point for this component
    reference =
      reference_time || if :start in component_time_points, do: :start, else: List.first(component_time_points)

    # Initialize assignments with reference point at 0
    assignments = %{reference => 0}

    # Assign other time points in this component based on constraints from reference
    Enum.reduce(component_time_points, assignments, fn time_point, acc ->
      if time_point != reference do
        # Find constraint from reference to this time point
        min_from_ref = find_min_constraint(constraints, reference, time_point)
        max_from_ref = find_max_constraint(constraints, reference, time_point)

        if min_from_ref != nil and max_from_ref != nil do
          # Assign a time within the constraint bounds
          assigned_time = (min_from_ref + max_from_ref) / 2
          Map.put(acc, time_point, assigned_time)
        else
          # No direct constraint from reference, assign 0 as fallback
          Map.put(acc, time_point, 0)
        end
      else
        acc
      end
    end)
  end

  # Helper functions for matrix operations
  @spec get_distance(distance_matrix(), time_point(), time_point(), number()) :: number()
  defp get_distance(matrix, from, to, default) do
    matrix
    |> Map.get(from, %{})
    |> Map.get(to, default)
  end

  @spec put_distance(distance_matrix(), time_point(), time_point(), number()) :: distance_matrix()
  defp put_distance(matrix, from, to, value) do
    from_map = Map.get(matrix, from, %{})
    updated_from_map = Map.put(from_map, to, value)
    Map.put(matrix, from, updated_from_map)
  end
end
