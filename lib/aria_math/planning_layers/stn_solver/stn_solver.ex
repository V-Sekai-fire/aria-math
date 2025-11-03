# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.StnSolver do
  @moduledoc """
  Axon-based Simple Temporal Network (STN) solver.

  This module implements constraint propagation and consistency checking
  using Axon tensor operations for GPU/CPU-accelerated temporal reasoning.

  ## Axon Model Architecture

  The STN solver uses an Axon model that processes constraint matrices through:
  - Constraint propagation layers
  - Consistency checking operations
  - Time assignment computation

  ## Integration Points

  This solver integrates with the temporal planning system through:
  - `AriaPlanner.Planner.Temporal.STN.Units.solve_stn/1` - for complex STN solving
  - `AriaPlanner.Planner.Temporal.STN.Consistency.mathematically_consistent?/1` - for consistency checking

  ## Algorithm Overview

  STNs represent temporal constraints between time points. The Floyd-Warshall algorithm
  is used to:
  1. Propagate constraints through the network (find all implied constraints)
  2. Detect negative cycles (inconsistent temporal constraints)
  3. Assign concrete time values to time points

  ## Submodules

  - `AriaMath.AxonLayers.StnSolver.Axon` - Neural network models and tensor operations
  - `AriaMath.AxonLayers.StnSolver.Core` - Core STN algorithms and data structures
  """

  alias AriaMath.AxonLayers.StnSolver.{Axon, Core}

  @type time_point :: atom()
  @type constraint :: {time_point(), time_point(), number(), number()}
  @type constraints :: [constraint()]
  @type consistency_result :: {:consistent, constraints()} | {:inconsistent, String.t()}

  @doc """
  Build an Axon model for STN constraint solving.

  Delegated to AriaMath.AxonLayers.StnSolver.Axon.build_model/1
  """
  @spec build_model(keyword()) :: Axon.t()
  defdelegate build_model(opts \\ []), to: Axon

  @doc """
  Check if a Simple Temporal Network is consistent using Floyd-Warshall algorithm.

  Delegated to AriaMath.AxonLayers.StnSolver.Core.check_consistency/1
  """
  @spec check_consistency(constraints()) :: consistency_result()
  defdelegate check_consistency(constraints), to: Core

  @doc """
  Solve STN constraints and return solved network.

  This is the main entry point that matches the MiniZinc interface.
  """
  @spec solve_stn(any()) :: {:ok, any()} | {:error, String.t()}
  def solve_stn(stn) do
    # Convert STN format to our internal constraint format
    constraints = Core.stn_to_constraints(stn)

    case Core.check_consistency(constraints) do
      {:consistent, tightened_constraints} ->
        # Convert back to STN format with tightened constraints
        solved_stn = Core.constraints_to_stn(stn, tightened_constraints)
        {:ok, solved_stn}

      {:inconsistent, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Assign concrete time values to time points in a consistent STN.

  Delegated to AriaMath.AxonLayers.StnSolver.Core.assign_times/2
  """
  @spec assign_times(constraints(), time_point() | nil) ::
          {:ok, %{time_point() => number()}} | {:error, String.t()}
  defdelegate assign_times(constraints, reference_time \\ nil), to: Core
end
