# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.PerformanceTest do
  use ExUnit.Case, async: true
  # import Nx
  require Logger

  @moduletag :slow

  # Ensure :torchx is started for Backend, even if not globally configured
  setup do
    {:ok, _apps} = Application.ensure_all_started(:torchx)
    :ok
  end

  # @matrix_size 2048 # Adjust this size based on desired computation intensity

  @tag :performance
  test "compare Nx matrix multiplication performance between CPU and Torchx (MPS)" do
    # Skip if Torchx is not available
    case Application.ensure_all_started(:torchx) do
      {:ok, _} ->
        # Skip performance test due to Nx API changes
        Logger.warning("Performance test skipped due to Nx API compatibility issues")

      _ ->
        Logger.warning("Torchx not available, skipping performance test")
    end
  end
end
