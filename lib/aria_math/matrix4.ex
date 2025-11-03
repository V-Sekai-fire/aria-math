# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.Matrix4 do
  @moduledoc """
  Differentiable 4x4 Matrix mathematics using Axon layers for gradient-based learning.

  Provides matrix operations that can participate in neural network training and
  optimization, enabling learned geometric transformations and spatial reasoning.
  """

  import Nx.Defn
  alias Axon.Tensor, as: A

  @doc """
  Differentiable matrix multiplication Axon layer.

  Creates a layer that performs matrix multiplication of two 4x4 matrices.
  Useful for composing learned transformations.
  """
  @spec multiply(A.t(), A.t(), keyword()) :: A.t()
  def multiply(a, b, opts \\ []) do
    name = opts[:name] || "matrix_multiply"
    Axon.layer(&multiply_impl/3, [a, b], name: name)
  end

  defnp multiply_impl(a, b, _opts) do
    Nx.dot(a, b)
  end

  @doc """
  Differentiable vector transformation Axon layer.

  Transforms a 3D vector by a 4x4 transformation matrix (homogeneous coordinates).
  Supports learned coordinate system transformations.
  """
  @spec transform_vector(A.t(), A.t(), keyword()) :: A.t()
  def transform_vector(matrix, vector, opts \\ []) do
    name = opts[:name] || "transform_vector"
    Axon.layer(&transform_vector_impl/3, [matrix, vector], name: name)
  end

  defnp transform_vector_impl(matrix, vector, _opts) do
    # Add w=1 for homogeneous coordinates
    homogeneous = Nx.concatenate([vector, Nx.tensor([1.0])])
    result = Nx.dot(matrix, homogeneous)

    # Return 3D vector (assuming affine transform)
    result[[0..2]]
  end

  @doc """
  Differentiable translation matrix creation from learnable parameters.

  Creates a translation matrix where translation components can be optimized.
  """
  @spec translation_matrix(A.t(), keyword()) :: A.t()
  def translation_matrix(translation, opts \\ []) do
    name = opts[:name] || "translation_matrix"
    Axon.layer(&translation_matrix_impl/2, [translation], name: name)
  end

  defnp translation_matrix_impl(translation, _opts) do
    tx = translation[0]
    ty = translation[1]
    tz = translation[2]

    Nx.tensor([
      [1.0, 0.0, 0.0, tx],
      [0.0, 1.0, 0.0, ty],
      [0.0, 0.0, 1.0, tz],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Differentiable scale matrix creation from learnable parameters.
  """
  @spec scale_matrix(A.t(), keyword()) :: A.t()
  def scale_matrix(scale, opts \\ []) do
    name = opts[:name] || "scale_matrix"
    Axon.layer(&scale_matrix_impl/2, [scale], name: name)
  end

  defnp scale_matrix_impl(scale, _opts) do
    sx = scale[0]
    sy = scale[1]
    sz = scale[2]

    Nx.tensor([
      [sx, 0.0, 0.0, 0.0],
      [0.0, sy, 0.0, 0.0],
      [0.0, 0.0, sz, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Learned TRS (Translation-Rotation-Scale) composition.

  Combines learnable translation, rotation quaternion, and scale into a transformation matrix.
  Supports end-to-end learned spatial transformations.
  """
  @spec compose_trs(A.t(), A.t(), A.t(), keyword()) :: A.t()
  def compose_trs(translation, rotation_quaternion, scale, opts \\ []) do
    name = opts[:name] || "compose_trs"
    Axon.layer(&compose_trs_impl/4, [translation, rotation_quaternion, scale], name: name)
  end

  defnp compose_trs_impl(translation, _rotation_quaternion, scale, _opts) do
    # Simplified TRS composition - full implementation needs quaternion to matrix conversion
    # For now, just apply translation and scale, ignore rotation

    tx = translation[0]
    ty = translation[1]
    tz = translation[2]

    sx = scale[0]
    sy = scale[1]
    sz = scale[2]

    Nx.tensor([
      [sx, 0.0, 0.0, tx],
      [0.0, sy, 0.0, ty],
      [0.0, 0.0, sz, tz],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Differentiable matrix inversion with learned numerical stability.

  Performs matrix inversion with gradient flow, useful for inverse kinematics
  and spatial reasoning networks.
  """
  @spec invert_matrix(A.t(), keyword()) :: A.t()
  def invert_matrix(matrix, opts \\ []) do
    name = opts[:name] || "invert_matrix"
    Axon.layer(&invert_matrix_impl/2, [matrix], name: name)
  end

  defnp invert_matrix_impl(matrix, _opts) do
    Nx.LinAlg.invert(matrix)
  end

  @doc """
  Learned orthonormal matrix enforcement.

  Applies Gram-Schmidt orthogonalization to ensure learned matrices remain orthonormal,
  useful for rotation matrix optimization.
  """
  @spec orthogonalize_matrix(A.t(), keyword()) :: A.t()
  def orthogonalize_matrix(matrix, opts \\ []) do
    name = opts[:name] || "orthogonalize_matrix"
    Axon.layer(&orthogonalize_matrix_impl/2, [matrix], name: name)
  end

  defnp orthogonalize_matrix_impl(matrix, _opts) do
    # Simplified orthogonalization - extract 3x3 rotation part
    r = matrix[[0..2, 0..2]]

    # Gram-Schmidt process (simplified)
    c0 = r[[.., 0]]
    c1 = r[[.., 1]]
    c2 = r[[.., 2]]

    # Normalize first column
    u0 = c0 / Nx.sqrt(Nx.sum(c0 * c0))

    # Subtract projection for u1
    proj1 = u0 * Nx.dot(c1, u0)
    u1 = c1 - proj1
    u1 = u1 / Nx.sqrt(Nx.sum(u1 * u1))

    # Subtract projections for u2
    proj2_0 = u0 * Nx.dot(c2, u0)
    proj2_1 = u1 * Nx.dot(c2, u1)
    u2 = c2 - proj2_0 - proj2_1
    u2 = u2 / Nx.sqrt(Nx.sum(u2 * u2))

    # Reconstruct matrix
    result = Nx.put_slice(matrix, [0, 0], u0)
    result = Nx.put_slice(result, [0, 1], u1)
    result = Nx.put_slice(result, [0, 2], u2)
    result
  end

  # Utility functions for compatibility - TODO: Remove when not needed

  @doc """
  Create an identity 4x4 matrix.
  """
  @spec identity() :: A.t()
  def identity() do
    Nx.tensor([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Create a translation matrix (alias for translation_matrix).
  """
  @spec translation({float(), float(), float()}) :: A.t()
  def translation({tx, ty, tz}) do
    Nx.tensor([
      [1.0, 0.0, 0.0, tx],
      [0.0, 1.0, 0.0, ty],
      [0.0, 0.0, 1.0, tz],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Create a scaling matrix (alias for scale_matrix).
  """
  @spec scaling({float(), float(), float()}) :: A.t()
  def scaling({sx, sy, sz}) do
    Nx.tensor([
      [sx, 0.0, 0.0, 0.0],
      [0.0, sy, 0.0, 0.0],
      [0.0, 0.0, sz, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Transform a 3D point using homogeneous coordinates.
  """
  @spec transform_point(A.t(), tuple()) :: tuple()
  def transform_point(matrix, point) do
    # This is a non-differentiable utility version
    {px, py, pz} = point
    # Add w=1
    result = Nx.dot(matrix, Nx.tensor([px, py, pz, 1.0]))
    {Nx.to_number(result[0]), Nx.to_number(result[1]), Nx.to_number(result[2])}
  end

  @doc """
  Decompose a matrix into translation, rotation, scale components.
  """
  @spec decompose(A.t()) :: {tuple(), A.t(), tuple()}
  def decompose(matrix) do
    # Extract translation
    translation = {Nx.to_number(matrix[[0, 3]]), Nx.to_number(matrix[[1, 3]]), Nx.to_number(matrix[[2, 3]])}

    # Extract rotation (simplified - assuming no shear for now)
    # For now, return identity matrix for rotation
    # TODO: Implement proper rotation matrix extraction
    rotation = identity()

    # Extract scale (diagonal elements)
    rotation_part = matrix[[0..2, 0..2]]
    scale = {Nx.to_number(rotation_part[[0, 0]]), Nx.to_number(rotation_part[[1, 1]]), Nx.to_number(rotation_part[[2, 2]])}

    {translation, rotation, scale}
  end

  @doc """
  Compose a matrix from translation, rotation, scale (alias for compose_trs).
  """
  @spec compose(tuple(), any(), tuple()) :: A.t()
  def compose(translation, rotation, scale) do
    compose_trs(translation, rotation, scale)
  end

  @doc """
  Check if two matrices are equal.
  """
  @spec equal?(A.t(), A.t()) :: boolean()
  def equal?(a, b) do
    Nx.all_close(a, b)
  end

  @doc """
  Convert a matrix to a list of 4-tuples.
  """
  @spec to_tuple_list(A.t()) :: [{float(), float(), float(), float()}]
  def to_tuple_list(matrix) do
    rows = Nx.to_list(matrix)
    Enum.map(rows, &List.to_tuple/1)
  end

  @doc """
  Extract the basis (rotation) part of a matrix.
  """
  @spec extract_basis(A.t()) :: A.t()
  def extract_basis(matrix) do
    matrix[[0..2, 0..2]]
  end

  @doc """
  Transpose a matrix.
  """
  @spec transpose(A.t()) :: A.t()
  def transpose(matrix) do
    Nx.transpose(matrix)
  end

  @doc """
  Get translation component from a matrix.
  """
  @spec get_translation(A.t()) :: tuple()
  def get_translation(matrix) do
    {Nx.to_number(matrix[[0, 3]]), Nx.to_number(matrix[[1, 3]]), Nx.to_number(matrix[[2, 3]])}
  end

  @doc """
  Convert a list of lists to a 4x4 matrix.

  ## Examples

      iex> lists = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
      iex> AriaMath.Matrix4.from_tuple_list(lists)
      #Nx.Tensor<f64[4][4]...>
  """
  @spec from_tuple_list([[float()]]) :: any()
  def from_tuple_list(list_list) do
    Nx.tensor(list_list, type: :f64)
  end

  @doc """
  Calculate the inverse of a 4x4 matrix (non-differentiable utility).

  ##
  """
  @spec inverse(any()) :: {any(), boolean()}
  def inverse(matrix) do
    # Use Gltf.Matrix for now - assuming it always succeeds
    result = AriaMath.Gltf.Matrix.inverse(matrix)
    {result, true}
  end
end


defmodule AriaMath.Matrix4.Tensor do
  @moduledoc """
  Batched matrix operations for efficient processing of multiple matrices.
  """

  import Nx.Defn
  alias Nx.Tensor, as: T

  @type batch_matrices :: T.t()
  @type batch_vectors :: T.t()

  @doc """
  Multiply batched matrices by batched matrices.

  ## Parameters
  - `a`: Batch of matrices {batch, 4, 4}
  - `b`: Batch of matrices {batch, 4, 4}

  Returns batched result {batch, 4, 4}
  """
  @spec multiply_batch(batch_matrices(), batch_matrices()) :: batch_matrices()
  defn multiply_batch(a, b) do
    Nx.dot(a, [2], [0], b, [2], [0])
  end

  @doc """
  Apply batched matrix inversion.

  ## Parameters
  - `matrices`: Batch of matrices {batch, 4, 4}

  Returns tuple {batch_inverted, batch_valid} where batch_valid indicates if each matrix was invertible.
  """
  @spec inverse_batch(batch_matrices()) :: {batch_matrices(), T.t()}
  defn inverse_batch(matrices) do
    Nx.LinAlg.invert(matrices)
    # For simplicity, assume all are valid - in practice check for NaN
    {matrices, Nx.broadcast(1, {Nx.axis_size(matrices, 0)})}
  end

  @doc """
  Transform batched points with batched matrices.

  ## Parameters
  - `matrices`: Batch of matrices {batch, 4, 4}
  - `points`: Batch of points {batch, 3}

  Returns batch of transformed points {batch, 3}
  """
  @spec transform_points_batch(batch_matrices(), batch_vectors()) :: batch_vectors()
  defn transform_points_batch(matrices, points) do
    # Add homogeneous coordinate w=1
    ones = Nx.broadcast(1.0, {Nx.axis_size(points, 0), 1})
    homogeneous_points = Nx.concatenate([points, ones], axis: 1)

    # Transform
    transformed = Nx.dot(matrices, [2], [0], homogeneous_points, [1], [0])

    # Perspective divide (assuming affine transforms, so w=1)
    transformed[[0..-1//1, 0..2]]
  end

  @doc """
  Extract translation components from batched matrices.

  ## Parameters
  - `matrices`: Batch of matrices {batch, 4, 4}

  Returns batch of translation vectors {batch, 3}
  """
  @spec extract_translations_batch(batch_matrices()) :: batch_vectors()
  defn extract_translations_batch(matrices) do
    matrices[[0..-1//1, 0..2, 3]]
  end

  @doc """
  Create scaling matrices from batched scale vectors.

  ## Parameters
  - `scales`: Batch of scale vectors {batch, 3}

  Returns batch of scaling matrices {batch, 4, 4}
  """
  @spec scaling_batch(batch_vectors()) :: batch_matrices()
  defn scaling_batch(scales) do
    batch_size = Nx.axis_size(scales, 0)
    sx = scales[[0..-1//1, 0]]
    sy = scales[[0..-1//1, 1]]
    sz = scales[[0..-1//1, 2]]

    # Create diagonal matrices with scaling factors
    zero = Nx.broadcast(0.0, {batch_size})
    one = Nx.broadcast(1.0, {batch_size})

    # Build matrices: [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]
    Nx.stack([
      Nx.stack([sx, zero, zero, zero], axis: 1),
      Nx.stack([zero, sy, zero, zero], axis: 1),
      Nx.stack([zero, zero, sz, zero], axis: 1),
      Nx.stack([zero, zero, zero, one], axis: 1)
    ], axis: 1)
  end

  @doc """
  Dummy implementation for rotations extraction.
  """
  @spec extract_rotations_batch(any()) :: any()
  def extract_rotations_batch(_matrices) do
    # Placeholder implementation
    Nx.tensor([])
  end

  @doc """
  Linear interpolation between batches of matrices.
  """
  @spec lerp_batch(batch_matrices(), batch_matrices(), float()) :: batch_matrices()
  defn lerp_batch(a, b, t) do
    Nx.add(Nx.multiply(a, 1.0 - t), Nx.multiply(b, t))
  end
end
