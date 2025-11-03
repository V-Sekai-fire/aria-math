# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.MatrixFixtures do
  @moduledoc """
  Golden standard fixtures for Matrix4 operations.

  These fixtures define expected results for matrix operations to ensure
  correctness without relying on debug logging.
  """

  alias AriaMath.Matrix4

  @doc """
  Identity matrix fixture.
  """
  def identity do
    Matrix4.from_tuple_list([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Translation matrix fixture: translate by (1, 2, 3).
  """
  def translation_1_2_3 do
    Matrix4.from_tuple_list([
      [1.0, 0.0, 0.0, 1.0],
      [0.0, 1.0, 0.0, 2.0],
      [0.0, 0.0, 1.0, 3.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Scaling matrix fixture: scale by (2, 3, 4).
  """
  def scaling_2_3_4 do
    Matrix4.from_tuple_list([
      [2.0, 0.0, 0.0, 0.0],
      [0.0, 3.0, 0.0, 0.0],
      [0.0, 0.0, 4.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Combined transform: translation(1,2,3) * scaling(2,3,4).
  """
  def translation_then_scaling do
    Matrix4.from_tuple_list([
      [2.0, 0.0, 0.0, 1.0],
      [0.0, 3.0, 0.0, 2.0],
      [0.0, 0.0, 4.0, 3.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Inverse of translation(1,2,3) should be translation(-1,-2,-3).
  """
  def inverse_translation_1_2_3 do
    Matrix4.from_tuple_list([
      [1.0, 0.0, 0.0, -1.0],
      [0.0, 1.0, 0.0, -2.0],
      [0.0, 0.0, 1.0, -3.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Point transformation fixtures.
  """
  def point_transforms do
    %{
      # Transform point (1,1,1) by identity -> should remain (1,1,1)
      identity_point: {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}},

      # Transform point (0,0,0) by translation(1,2,3) -> should be (1,2,3)
      translation_origin: {{0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}},

      # Transform point (1,1,1) by translation(1,2,3) -> should be (2,3,4)
      translation_point: {{1.0, 1.0, 1.0}, {2.0, 3.0, 4.0}}
    }
  end

  @doc """
  Matrix decomposition fixtures.
  """
  def decompositions do
    %{
      # Identity: translation(0,0,0), identity rotation, scale(1,1,1)
      identity: {
        {0.0, 0.0, 0.0},
        identity(),
        {1.0, 1.0, 1.0}
      },

      # Translation only: translation(1,2,3), identity rotation, scale(1,1,1)
      translation_only: {
        {1.0, 2.0, 3.0},
        identity(),
        {1.0, 1.0, 1.0}
      },

      # Scaling only: translation(0,0,0), identity rotation, scale(2,3,4)
      scaling_only: {
        {0.0, 0.0, 0.0},
        identity(),
        {2.0, 3.0, 4.0}
      }
    }
  end

  @doc """
  Transform hierarchy fixtures for Joint tests.
  """
  def transform_hierarchies do
    %{
      # Parent at (1,0,0), child at (0,1,0) relative to parent
      # Expected global child position: (1,1,0)
      parent_child_simple: %{
        parent_transform: translation_1_0_0(),
        child_local_transform: translation_0_1_0(),
        expected_global_child: translation_1_1_0()
      },

      # Root at origin, child at (1,2,3)
      root_child: %{
        parent_transform: identity(),
        child_local_transform: translation_1_2_3(),
        expected_global_child: translation_1_2_3()
      }
    }
  end

  # Helper matrices
  defp translation_1_0_0 do
    Matrix4.from_tuple_list([
      [1.0, 0.0, 0.0, 1.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  defp translation_0_1_0 do
    Matrix4.from_tuple_list([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 1.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  defp translation_1_1_0 do
    Matrix4.from_tuple_list([
      [1.0, 0.0, 0.0, 1.0],
      [0.0, 1.0, 0.0, 1.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end
end
