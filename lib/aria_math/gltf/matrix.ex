# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.Gltf.Matrix do
  @moduledoc """
  GLTF Interactivity Extension compliant 4x4 Matrix mathematics implementation
  using Nx for numerical operations.

  This module provides matrix operations as defined in the glTF Interactivity Extension
  Specification, leveraging Nx for efficient and robust calculations.
  """

  import Nx.Defn
  alias Nx.LinAlg
  alias Nx.Tensor, as: T

  @type t :: T.t()

  @doc """
  Calculates the transpose of a matrix.
  Corresponds to `math/transpose` in glTF Interactivity Extension.
  """
  @spec transpose(t()) :: t()
  defn transpose(matrix) do
    Nx.transpose(matrix)
  end

  @doc """
  Calculates the determinant of a matrix.
  Corresponds to `math/determinant` in glTF Interactivity Extension.
  """
  @spec determinant(t()) :: float()
  defn determinant(matrix) do
    LinAlg.determinant(matrix)
  end

  @doc """
  Calculates the inverse of a matrix.
  Corresponds to `math/inverse` in glTF Interactivity Extension.
  Returns the identity matrix if the matrix is not invertible.
  """
  @spec inverse(t()) :: t()
  defn inverse(matrix) do
    inv_matrix = LinAlg.invert(matrix)

    # Nx.LinAlg.invert returns NaN for singular matrices.
    # We return identity() in such cases, as per glTF spec's implicit behavior for invalid inverse.
    if Nx.any(Nx.is_nan(inv_matrix)) do
      identity()
    else
      inv_matrix
    end
  end

  @doc """
  Performs matrix multiplication.
  Corresponds to `math/matMul` in glTF Interactivity Extension.
  """
  @spec mat_mul(t(), t()) :: t()
  defn mat_mul(a, b) do
    Nx.dot(a, b)
  end

  # Helper function for identity matrix, used by inverse/1
  defn identity() do
    Nx.tensor([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ], type: :f64)
  end
end
