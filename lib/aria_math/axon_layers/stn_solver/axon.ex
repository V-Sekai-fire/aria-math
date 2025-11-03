# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.StnSolver.Axon do
  @moduledoc """
  Axon-specific tensor operations for STN solver.

  This module contains the neural network model definitions
  and tensor-based operations for STN constraint solving.
  """



  @doc """
  Build an Axon model for STN constraint solving.

  ## Parameters
  - `opts` - Model configuration options
    - `:max_timepoints` - Maximum number of time points (default: 10)

  ## Returns
  - Axon model that can process constraint matrices and output solved STN

  ## Usage
  ```elixir
  model = StnSolver.Axon.build_model(max_timepoints: 8)
  params = Axon.init(model)
  # Use for prediction or training
  ```
  """
  @spec build_model(keyword()) :: Axon.t()
  def build_model(opts \\ []) do
    max_timepoints = Keyword.get(opts, :max_timepoints, 10)

    # Input: constraint matrix representation
    # Shape: [batch_size, max_timepoints, max_timepoints, 2] (min/max constraints)
    constraint_input = Axon.input("constraints", shape: {nil, max_timepoints, max_timepoints, 2})

    # Extract min and max constraint matrices
    min_constraints = Axon.slice(constraint_input, [0, 0, 0, 0], [nil, max_timepoints, max_timepoints, 1])
    max_constraints = Axon.slice(constraint_input, [0, 0, 0, 1], [nil, max_timepoints, max_timepoints, 1])

    # Squeeze to remove single dimension
    min_constraints = Axon.squeeze(min_constraints, axes: [3])
    max_constraints = Axon.squeeze(max_constraints, axes: [3])

    # Apply constraint propagation layers
    propagated_min = constraint_propagation_layer(min_constraints, max_timepoints)
    propagated_max = constraint_propagation_layer(max_constraints, max_timepoints)

    # Consistency checking output
    consistency_output = consistency_checking_layer(propagated_min, propagated_max)

    # Time assignment output
    time_assignment_output = time_assignment_layer(propagated_min, propagated_max)

    # Combine outputs
    Axon.container(%{
      consistency: consistency_output,
      time_assignments: time_assignment_output,
      min_constraints: propagated_min,
      max_constraints: propagated_max
    })
  end

  @doc """
  Constraint propagation layer - simplified differentiable version of constraint propagation.
  """
  @spec constraint_propagation_layer(Axon.t(), non_neg_integer()) :: Axon.t()
  def constraint_propagation_layer(constraint_matrix, _max_timepoints) do
    # Apply relaxation-based constraint propagation
    # This is a simplified version - real implementation would need more sophisticated constraint propagation

    # Add identity matrix for reflexive constraints (distance to self is 0)
    identity = Axon.eye(constraint_matrix, key: :identity)
    constraint_matrix = Axon.add(constraint_matrix, identity)

    # Apply iterative relaxation (simplified constraint tightening)
    # In practice, this would be more complex with proper Floyd-Warshall or similar
    constraint_matrix
    |> Axon.dense(128, name: "prop1")
    |> Axon.relu()
    |> Axon.dense(elem(constraint_matrix.shape, 2), name: "prop2")
  end

  @doc """
  Consistency checking layer.
  """
  @spec consistency_checking_layer(Axon.t(), Axon.t()) :: Axon.t()
  def consistency_checking_layer(min_matrix, max_matrix) do
    # Check if min <= max for all constraints
    diff_matrix = Axon.subtract(max_matrix, min_matrix)
    # Find negative differences (inconsistency)
    neg_mask = Axon.less(diff_matrix, 0.0)
    # Check if any negative (inconsistency exists)
    Axon.reduce_max(neg_mask, axes: [1, 2], keep_axes: true)
    |> Axon.squeeze(axes: [1, 2])
    |> Axon.logical_not()
  end

  @doc """
  Time assignment layer.
  """
  @spec time_assignment_layer(Axon.t(), Axon.t()) :: Axon.t()
  def time_assignment_layer(min_matrix, max_matrix) do
    # Simple time assignment: average of min/max bounds
    # More sophisticated would use proper STN time assignment
    average_bounds = Axon.divide(Axon.add(min_matrix, max_matrix), 2.0)

    # Set first timepoint as reference (time 0)
    # This is a simplified assignment - real STN assignment is more complex
    first_col = Axon.slice(average_bounds, [0, 0, 0], [nil, 1, nil])
    Axon.subtract(average_bounds, first_col)
  end
end
