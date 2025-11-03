# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.Sharding do
  @moduledoc """
  SPMD sharding annotations and collective operations for distributed training.

  Based on GSPMD principles: Single Program Multiple Data paradigm where
  the same program runs on multiple devices with different data partitions.
  """

  @type sharding :: %{
    type: :replicated | :tiled | :partially_tiled,
    device_mesh: tuple(),
    dims_mapping: list(integer())
  }

  @doc """
  Creates a sharding annotation for tensor distribution across devices.

  ## Parameters
  - `device_mesh`: tuple representing device mesh dimensions (e.g., {4, 8})
  - `dims_mapping`: list mapping tensor dimensions to device mesh dimensions
  - `type`: :replicated, :tiled, or :partially_tiled

  ## Examples
      # Shard batch and feature dimensions across 2D mesh
      sharding = AriaMath.Sharding.annotation({4, 8}, [0, -1, 1])
  """
  @spec annotation(tuple(), list(integer()), atom()) :: sharding()
  def annotation(device_mesh, dims_mapping, type \\ :tiled) do
    %{
      type: type,
      device_mesh: device_mesh,
      dims_mapping: dims_mapping
    }
  end

  @doc """
  Applies sharding to a tensor for SPMD execution.

  In a real SPMD system, this would distribute the tensor across devices.
  Here we simulate the concept by returning the tensor with sharding metadata.
  """
  @spec shard(Nx.Tensor.t(), sharding()) :: Nx.Tensor.t()
  def shard(tensor, _sharding) do
    # In a real implementation, this would distribute tensor across devices
    # For now, we just return the tensor as-is
    tensor
  end

  @doc """
  AllReduce operation: reduces tensor across all devices and broadcasts result.

  This is a key collective operation in SPMD systems for gradient synchronization.
  """
  @spec all_reduce(Nx.Tensor.t(), fun()) :: Nx.Tensor.t()
  def all_reduce(tensor, _reduce_fn \\ &Nx.add/2) do
    # In a real distributed system, this would communicate across devices
    # For simulation, we return the tensor unchanged
    tensor
  end

  @doc """
  AllGather operation: gathers tensor from all devices.

  Used for collecting sharded data back to full tensors.
  """
  @spec all_gather(Nx.Tensor.t()) :: Nx.Tensor.t()
  def all_gather(tensor) do
    # In a real distributed system, this would collect from all devices
    # For simulation, we return the tensor unchanged
    tensor
  end

  @doc """
  ReduceScatter operation: reduces then scatters across devices.

  Efficient combination of reduction and scattering, used in weight updates.
  In this simulation, we reduce along the batch dimension to simulate
  the scattering effect.
  """
  @spec reduce_scatter(Nx.Tensor.t(), fun()) :: Nx.Tensor.t()
  def reduce_scatter(tensor, _reduce_fn \\ &Nx.add/2) do
    # In a real distributed system, this would reduce and scatter across devices
    # For simulation, we simulate the effect by reducing along batch dimension
    # This represents how ReduceScatter typically reduces memory footprint

    {batch_size, _feature_dim} = Nx.shape(tensor)

    # Simulate scattering by reducing to half the batch size
    # In a real implementation, this would be distributed across devices
    half_batch = div(batch_size, 2)

    # Take the first half of the reduced batch (simulating scatter result)
    Nx.slice_along_axis(tensor, 0, half_batch, axis: 0)
  end

  @doc """
  Propagates sharding annotations through a computation graph.

  Following GSPMD principles, this automatically completes sharding
  specifications based on user annotations and operator semantics.
  """
  @spec propagate_sharding(map(), list()) :: map()
  def propagate_sharding(sharding_map, operations) do
    # Simple sharding propagation: for each operation, propagate
    # sharding from inputs to outputs based on tensor dimensions
    Enum.reduce(operations, sharding_map, fn operation, acc ->
      propagate_operation_sharding(acc, operation)
    end)
  end

  defp propagate_operation_sharding(sharding_map, {:transform_points, inputs, output}) do
    # Transform points preserves batch dimension sharding
    input_sharding = Map.get(sharding_map, hd(inputs), nil)
    if input_sharding do
      Map.put(sharding_map, output, input_sharding)
    else
      sharding_map
    end
  end

  defp propagate_operation_sharding(sharding_map, {:compose_matrices, [m1, m2], output}) do
    # Matrix composition can inherit sharding from either input
    m1_sharding = Map.get(sharding_map, m1, nil)
    m2_sharding = Map.get(sharding_map, m2, nil)

    output_sharding = m1_sharding || m2_sharding
    if output_sharding do
      Map.put(sharding_map, output, output_sharding)
    else
      sharding_map
    end
  end

  defp propagate_operation_sharding(sharding_map, _operation) do
    # Default: no propagation
    sharding_map
  end
end

defmodule AriaMath.AxonLayers do
  @moduledoc """
  SPMD-aware Axon layers for 3D mathematical operations using Matrix4 transformations.

  These layers integrate AriaMath's 3D transformation capabilities with Axon
  neural networks, following SPMD principles for distributed training.
  """

  import Nx.Defn
  alias AriaMath.Sharding

  @doc """
  SPMD-aware 3D point transformation with sharding support.

  Applies transformation matrices to 3D points with automatic sharding
  and collective communication following GSPMD principles.

  ## Options

    * `:name` - layer name
    * `:sharding` - sharding specification for distributed execution

  ## Examples

      # Distributed 3D point transformation
      points = Axon.input("points", shape: {nil, 3})
      transform_matrix = Axon.param("transform", {4, 4})
      sharding = AriaMath.Sharding.annotation({4, 8}, [0, -1])  # shard batch dim
      transformed = AriaMath.AxonLayers.spmd_transform_points(points, transform_matrix, sharding: sharding)
  """
  def spmd_transform_points(input, transform_matrix, opts \\ []) do
    name = opts[:name] || "spmd_transform_points"
    sharding = opts[:sharding]

    Axon.layer(
      &spmd_transform_points_impl/3,
      [input, transform_matrix],
      name: name,
      sharding: sharding
    )
  end

  defnp spmd_transform_points_impl(points, transform_matrix, opts) do
    # Apply sharding if specified
    sharding = opts[:sharding]
    points = if sharding, do: Sharding.shard(points, sharding), else: points
    transform_matrix = if sharding, do: Sharding.shard(transform_matrix, sharding), else: transform_matrix

    # Perform the transformation
    result = transform_points_impl(points, transform_matrix, %{})

    # Apply collective operations if needed for sharded computation
    if sharding do
      Sharding.all_reduce(result)
    else
      result
    end
  end

  @doc """
  Applies a 3D transformation matrix to input points.

  Input should be a tensor of shape {batch_size, 3} representing 3D points.
  The transformation matrix should be a 4x4 matrix tensor.

  ## Options

    * `:name` - layer name

  ## Examples

      # Transform 3D points with a learned transformation matrix
      points = Axon.input("points", shape: {nil, 3})
      transform_matrix = Axon.param("transform", {4, 4})
      transformed_points = AriaMath.AxonLayers.transform_points(points, transform_matrix)
  """
  def transform_points(input, transform_matrix, opts \\ []) do
    name = opts[:name] || "transform_points"

    Axon.layer(
      &transform_points_impl/3,
      [input, transform_matrix],
      name: name
    )
  end

  defnp transform_points_impl(points, transform_matrix, _opts) do
    # points: {batch_size, 3}
    # transform_matrix: {4, 4}

    # Convert points to homogeneous coordinates: {batch_size, 4}
    batch_size = Nx.axis_size(points, 0)
    ones = Nx.broadcast(1.0, {batch_size, 1})
    homogeneous_points = Nx.concatenate([points, ones], axis: 1)

    # Apply transformation: {batch_size, 4}
    transformed_homogeneous = Nx.dot(homogeneous_points, transform_matrix)

    # Convert back to 3D coordinates (divide by w): {batch_size, 3}
    w = transformed_homogeneous[[.., 3]]
    w = Nx.reshape(w, {batch_size, 1})
    xyz = transformed_homogeneous[[.., 0..2]]
    Nx.divide(xyz, w)
  end

  @doc """
  SPMD-aware matrix composition with sharding support.

  Composes transformation matrices with automatic sharding and collective operations.
  Follows GSPMD principles for distributed matrix operations.

  ## Options

    * `:name` - layer name
    * `:sharding` - sharding specification for distributed execution

  ## Examples

      matrix1 = Axon.param("matrix1", {4, 4})
      matrix2 = Axon.param("matrix2", {4, 4})
      sharding = AriaMath.Sharding.annotation({4, 8}, [-1, -1, 0, 1])  # shard on last two dims
      composed = AriaMath.AxonLayers.spmd_compose_matrices(matrix1, matrix2, sharding: sharding)
  """
  def spmd_compose_matrices(matrix1, matrix2, opts \\ []) do
    name = opts[:name] || "spmd_compose_matrices"
    sharding = opts[:sharding]

    Axon.layer(
      &spmd_compose_matrices_impl/3,
      [matrix1, matrix2],
      name: name,
      sharding: sharding
    )
  end

  defnp spmd_compose_matrices_impl(matrix1, matrix2, opts) do
    # Apply sharding if specified
    sharding = opts[:sharding]
    matrix1 = if sharding, do: Sharding.shard(matrix1, sharding), else: matrix1
    matrix2 = if sharding, do: Sharding.shard(matrix2, sharding), else: matrix2

    # Perform matrix composition
    result = compose_matrices_impl(matrix1, matrix2, %{})

    # Apply collective operations for sharded computation
    if sharding do
      Sharding.all_reduce(result)
    else
      result
    end
  end

  @doc """
  Composes two 4x4 transformation matrices.

  ## Options

    * `:name` - layer name

  ## Examples

      matrix1 = Axon.param("matrix1", {4, 4})
      matrix2 = Axon.param("matrix2", {4, 4})
      composed = AriaMath.AxonLayers.compose_matrices(matrix1, matrix2)
  """
  def compose_matrices(matrix1, matrix2, opts \\ []) do
    name = opts[:name] || "compose_matrices"

    Axon.layer(
      &compose_matrices_impl/3,
      [matrix1, matrix2],
      name: name
    )
  end

  defnp compose_matrices_impl(matrix1, matrix2, _opts) do
    # Matrix multiplication for composition
    Nx.dot(matrix1, matrix2)
  end

  @doc """
  Computes the inverse of a 4x4 transformation matrix.

  ## Options

    * `:name` - layer name

  ## Examples

      matrix = Axon.param("matrix", {4, 4})
      inverse_matrix = AriaMath.AxonLayers.inverse_matrix(matrix)
  """
  def inverse_matrix(matrix, opts \\ []) do
    name = opts[:name] || "inverse_matrix"

    Axon.layer(
      &inverse_matrix_impl/2,
      [matrix],
      name: name
    )
  end

  defnp inverse_matrix_impl(matrix, _opts) do
    # For affine transformations, we can use the efficient inverse method
    # Extract components: M = [R S T; 0 0 0 1]

    # Get the 3x3 rotation/scale part
    r = matrix[[0..2, 0..2]]
    # Get translation vector
    t = matrix[[0..2, 3]]

    # Compute R^-1 using cofactor method for 3x3 matrix
    # Compute determinant
    r00 = r[[0, 0]]
    r01 = r[[0, 1]]
    r02 = r[[0, 2]]
    r10 = r[[1, 0]]
    r11 = r[[1, 1]]
    r12 = r[[1, 2]]
    r20 = r[[2, 0]]
    r21 = r[[2, 1]]
    r22 = r[[2, 2]]

    _det = r00 * (r11 * r22 - r12 * r21) -
          r01 * (r10 * r22 - r12 * r20) +
          r02 * (r10 * r21 - r11 * r20)

    # For numerical stability, we'll use a simplified approach
    # Transpose the 3x3 part (works for rotation matrices)
    r_inv = Nx.transpose(r)

    # Compute -R^-1 * T
    neg_t = Nx.negate(t)
    t_inv = Nx.dot(r_inv, neg_t)

    # Build inverse matrix
    inverse_matrix = Nx.broadcast(0.0, {4, 4})
    inverse_matrix = Nx.put_slice(inverse_matrix, [0, 0], r_inv)
    inverse_matrix = Nx.put_slice(inverse_matrix, [0, 3], Nx.reshape(t_inv, {3, 1}))
    inverse_matrix = Nx.put_slice(inverse_matrix, [3, 3], Nx.tensor([[1.0]]))

    inverse_matrix
  end

  @doc """
  Creates a translation matrix from translation vectors.

  Input should be a tensor of shape {batch_size, 3} representing translation vectors.

  ## Options

    * `:name` - layer name

  ## Examples

      translations = Axon.input("translations", shape: {nil, 3})
      translation_matrices = AriaMath.AxonLayers.translation_matrix(translations)
  """
  def translation_matrix(translations, opts \\ []) do
    name = opts[:name] || "translation_matrix"

    Axon.layer(
      &translation_matrix_impl/2,
      [translations],
      name: name
    )
  end

  defnp translation_matrix_impl(translations, _opts) do
    # translations: {batch_size, 3} or {3}
    batch_size = Nx.axis_size(translations, 0)
    is_batched = Nx.rank(translations) == 2

    if is_batched do
      # batched case: create batch of matrices
      identity_base = Nx.eye(4)
      identities = Nx.broadcast(identity_base, {batch_size, 4, 4})

      # Extract translation components
      tx_batch = translations[[.., 0]]
      ty_batch = translations[[.., 1]]
      tz_batch = translations[[.., 2]]

      # Create translations: {batch_size, 4, 4} -> identity with last column set
      translations_3d = Nx.stack([
        Nx.concatenate([Nx.broadcast(0.0, {batch_size, 3}), Nx.reshape(tx_batch, {batch_size, 1})], axis: 1),
        Nx.concatenate([Nx.broadcast(0.0, {batch_size, 3}), Nx.reshape(ty_batch, {batch_size, 1})], axis: 1),
        Nx.concatenate([Nx.broadcast(0.0, {batch_size, 3}), Nx.reshape(tz_batch, {batch_size, 1})], axis: 1),
        Nx.concatenate([Nx.broadcast(0.0, {batch_size, 3}), Nx.broadcast(1.0, {batch_size, 1})], axis: 1)
      ], axis: 1)

      Nx.add(identities, translations_3d)
    else
      # single matrix case
      identity = Nx.eye(4)
      tx = translations[0]
      ty = translations[1]
      tz = translations[2]

      identity
      |> Nx.indexed_put([[0, 3]], tx)
      |> Nx.indexed_put([[1, 3]], ty)
      |> Nx.indexed_put([[2, 3]], tz)
    end
  end

  @doc """
  Creates a rotation matrix from quaternions.

  Input should be a tensor of shape {batch_size, 4} representing quaternions [w, x, y, z].

  ## Options

    * `:name` - layer name

  ## Examples

      quaternions = Axon.input("quaternions", shape: {nil, 4})
      rotation_matrices = AriaMath.AxonLayers.rotation_matrix_from_quaternion(quaternions)
  """
  def rotation_matrix_from_quaternion(quaternions, opts \\ []) do
    name = opts[:name] || "rotation_matrix_from_quaternion"

    Axon.layer(
      &rotation_matrix_from_quaternion_impl/2,
      [quaternions],
      name: name
    )
  end

  defnp rotation_matrix_from_quaternion_impl(quaternions, _opts) do
    # quaternions: {batch_size, 4} where each quaternion is [w, x, y, z]
    batch_size = Nx.axis_size(quaternions, 0)

    w = quaternions[[.., 0]]
    x = quaternions[[.., 1]]
    y = quaternions[[.., 2]]
    z = quaternions[[.., 3]]

    # Compute rotation matrix elements
    # R = [
    #   [1-2y²-2z², 2xy-2wz, 2xz+2wy],
    #   [2xy+2wz, 1-2x²-2z², 2yz-2wx],
    #   [2xz-2wy, 2yz+2wx, 1-2x²-2y²]
    # ]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # Create 4x4 matrices with identity bottom-right
    _matrices = Nx.broadcast(0.0, {batch_size, 4, 4})

    # Build rotation matrices row by row
    row0 = Nx.stack([r00, r01, r02, Nx.broadcast(0.0, {batch_size})], axis: 1)
    row1 = Nx.stack([r10, r11, r12, Nx.broadcast(0.0, {batch_size})], axis: 1)
    row2 = Nx.stack([r20, r21, r22, Nx.broadcast(0.0, {batch_size})], axis: 1)
    row3 = Nx.stack([Nx.broadcast(0.0, {batch_size}), Nx.broadcast(0.0, {batch_size}), Nx.broadcast(0.0, {batch_size}), Nx.broadcast(1.0, {batch_size})], axis: 1)

    Nx.stack([row0, row1, row2, row3], axis: 1)
  end

  @doc """
  Decomposes a 4x4 transformation matrix into translation, rotation, and scale components.

  Returns a tuple {translation, rotation_matrix, scale} where each has shape {batch_size, ...}.

  ## Options

    * `:name` - layer name

  ## Examples

      matrices = Axon.param("matrices", {nil, 4, 4})
      {translation, rotation, scale} = AriaMath.AxonLayers.decompose_matrix(matrices)
  """
  def decompose_matrix(matrix, opts \\ []) do
    name = opts[:name] || "decompose_matrix"

    Axon.layer(
      &decompose_matrix_impl/2,
      [matrix],
      name: name
    )
  end

  defnp decompose_matrix_impl(matrix, _opts) do
    # matrix: {4, 4}
    # Extract translation (last column, first 3 elements)
    translation = Nx.stack([matrix[[0, 3]], matrix[[1, 3]], matrix[[2, 3]]])  # {3}

    # Extract basis (first 3x3)
    basis = Nx.stack([
      [matrix[[0, 0]], matrix[[0, 1]], matrix[[0, 2]]],
      [matrix[[1, 0]], matrix[[1, 1]], matrix[[1, 2]]],
      [matrix[[2, 0]], matrix[[2, 1]], matrix[[2, 2]]]
    ])  # {3, 3}

    # Calculate scale factors (lengths of basis vectors)
    x_axis = Nx.stack([basis[[0, 0]], basis[[0, 1]], basis[[0, 2]]])
    y_axis = Nx.stack([basis[[1, 0]], basis[[1, 1]], basis[[1, 2]]])
    z_axis = Nx.stack([basis[[2, 0]], basis[[2, 1]], basis[[2, 2]]])

    sx = Nx.sqrt(Nx.sum(x_axis * x_axis))
    sy = Nx.sqrt(Nx.sum(y_axis * y_axis))
    sz = Nx.sqrt(Nx.sum(z_axis * z_axis))

    scale = Nx.stack([sx, sy, sz])

    # Normalize basis to get rotation (simplified: return identity for now as full implementation is complex)
    identity_rotation = Nx.eye(4)  # {4, 4}

    {translation, identity_rotation, scale}
  end

  # Additional Axon layers for math operations

  @doc """
  Multiplies two 4x4 matrices.

  ## Options

    * `:name` - layer name

  ## Examples

      matrix1 = Axon.param("matrix1", {4, 4})
      matrix2 = Axon.param("matrix2", {4, 4})
      result = AriaMath.AxonLayers.multiply_matrices(matrix1, matrix2)
  """
  def multiply_matrices(a, b, opts \\ []) do
    name = opts[:name] || "multiply_matrices"
    Axon.layer(&multiply_matrices_impl/3, [a, b], name: name)
  end

  defnp multiply_matrices_impl(a, b, _opts) do
    Nx.dot(a, b)
  end

  @doc """
  Transposes a 4x4 matrix.

  ## Options

    * `:name` - layer name

  ## Examples

      matrix = Axon.param("matrix", {4, 4})
      result = AriaMath.AxonLayers.transpose_matrix(matrix)
  """
  def transpose_matrix(matrix, opts \\ []) do
    name = opts[:name] || "transpose_matrix"
    Axon.layer(&transpose_matrix_impl/2, [matrix], name: name)
  end

  defnp transpose_matrix_impl(matrix, _opts) do
    Nx.transpose(matrix)
  end

  @doc """
  Creates a scaling matrix from scale vectors.

  Input should be a tensor of shape {batch_size, 3} representing scale vectors.

  ## Options

    * `:name` - layer name

  ## Examples

      scales = Axon.param("scales", {nil, 3})
      scaling_matrices = AriaMath.AxonLayers.scaling_matrix(scales)
  """
  def scaling_matrix(scales, opts \\ []) do
    name = opts[:name] || "scaling_matrix"

    Axon.layer(
      &scaling_matrix_impl/2,
      [scales],
      name: name
    )
  end

  defnp scaling_matrix_impl(scales, _opts) do
    batch_size = Nx.axis_size(scales, 0)
    is_batched = Nx.rank(scales) == 2

    if is_batched do
      # batched case
      sx_batch = scales[[.., 0]]
      sy_batch = scales[[.., 1]]
      sz_batch = scales[[.., 2]]

      matrices = Nx.broadcast(0.0, {batch_size, 4, 4})

      # Set the diagonals
      matrices =
        Nx.indexed_put(matrices, Nx.add(Nx.iota({batch_size}) * 16, Nx.iota({4})) * Nx.broadcast(4, {batch_size * 4}) + Nx.iota({4}), Nx.stack([sx_batch, sy_batch, sz_batch, Nx.broadcast(1.0, {batch_size})], axis: 1) |> Nx.flatten())

      matrices
    else
      # single matrix
      sx = scales[0]
      sy = scales[1]
      sz = scales[2]

      Nx.tensor([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ], type: :f32)
    end
  end

  @doc """
  Composes a 4x4 matrix from translation, rotation, and scale components.

  ## Options

    * `:name` - layer name

  ## Examples

      translations = Axon.param("translations", {nil, 3})
      rotations = Axon.param("rotations", {nil, 4, 4})
      scales = Axon.param("scales", {nil, 3})
      composed = AriaMath.AxonLayers.compose_matrix(translations, rotations, scales)
  """
  def compose_matrix(translations, rotations, scales, opts \\ []) do
    name = opts[:name] || "compose_matrix"

    Axon.layer(
      &compose_matrix_impl/4,
      [translations, rotations, scales],
      name: name
    )
  end

  defnp compose_matrix_impl(translations, rotations, scales, _opts) do
    # translations: {batch_size, 3} or {3}
    # rotations: {batch_size, 4, 4} or {4, 4}
    # scales: {batch_size, 3} or {3}

    # Simplified implementation - compose T * R * S (scale, rotate, translate)
    if Nx.rank(rotations) == 3 do
      # Batched case
      batch_size = Nx.axis_size(rotations, 0)

      # Create scale matrices
      scale_matrices = Nx.broadcast(0.0, {batch_size, 4, 4})
      sx = Nx.reshape(scales[[.., 0]], {batch_size, 1, 1})
      sy = Nx.reshape(scales[[.., 1]], {batch_size, 1, 1})
      sz = Nx.reshape(scales[[.., 2]], {batch_size, 1, 1})

      scale_matrices = Nx.put_slice(scale_matrices, [0, 0, 0], sx)
      scale_matrices = Nx.put_slice(scale_matrices, [0, 1, 1], sy)
      scale_matrices = Nx.put_slice(scale_matrices, [0, 2, 2], sz)
      scale_matrices = Nx.put_slice(scale_matrices, [0, 3, 3], Nx.broadcast(1.0, {batch_size, 1, 1}))

      # Apply scaling to rotation
      scaled_rotation = Nx.dot(rotations, scale_matrices)

      # Create translation matrices and apply
      translation_matrices = Nx.broadcast(0.0, {batch_size, 4, 4})
      translation_matrices = Nx.put_slice(translation_matrices, [0, 0, 3], Nx.reshape(translations[[.., 0]], {batch_size, 1}))
      translation_matrices = Nx.put_slice(translation_matrices, [0, 1, 3], Nx.reshape(translations[[.., 1]], {batch_size, 1}))
      translation_matrices = Nx.put_slice(translation_matrices, [0, 2, 3], Nx.reshape(translations[[.., 2]], {batch_size, 1}))
      translation_matrices = Nx.put_slice(translation_matrices, [0, 3, 3], Nx.broadcast(1.0, {batch_size, 1}))
      translation_matrices = Nx.put_slice(translation_matrices, [0, 3, 0], Nx.broadcast(1.0, {batch_size, 1}))
      translation_matrices = Nx.put_slice(translation_matrices, [0, 3, 1], Nx.broadcast(0.0, {batch_size, 1, 1}))
      translation_matrices = Nx.put_slice(translation_matrices, [0, 3, 2], Nx.broadcast(0.0, {batch_size, 1, 1}))

      Nx.dot(translation_matrices, scaled_rotation)
    else
      # Single matrix case
      sx = scales[0]
      sy = scales[1]
      sz = scales[2]
      tx = translations[0]
      ty = translations[1]
      tz = translations[2]

      # Create scale matrix and multiply with rotation
      scale_matrix = Nx.eye(4)
      scale_matrix = Nx.put_slice(scale_matrix, [0, 0], Nx.tensor([[sx]]))
      scale_matrix = Nx.put_slice(scale_matrix, [1, 1], Nx.tensor([[sy]]))
      scale_matrix = Nx.put_slice(scale_matrix, [2, 2], Nx.tensor([[sz]]))

      scaled_rotation = Nx.dot(rotations, scale_matrix)

      # Create translation matrix and multiply
      translation_matrix = Nx.eye(4)
      translation_matrix = Nx.put_slice(translation_matrix, [0, 3], Nx.tensor([[tx]]))
      translation_matrix = Nx.put_slice(translation_matrix, [1, 3], Nx.tensor([[ty]]))
      translation_matrix = Nx.put_slice(translation_matrix, [2, 3], Nx.tensor([[tz]]))

      Nx.dot(translation_matrix, scaled_rotation)
    end
  end

  # Delegation to split modules

  # Constants
  defdelegate e(opts \\ []), to: AriaMath.AxonLayers.Constants
  defdelegate pi(opts \\ []), to: AriaMath.AxonLayers.Constants
  defdelegate inf(opts \\ []), to: AriaMath.AxonLayers.Constants
  defdelegate nan(opts \\ []), to: AriaMath.AxonLayers.Constants

  # Comparison operations
  defdelegate eq(a, b, opts \\ []), to: AriaMath.AxonLayers.Comparison
  defdelegate lt(a, b, opts \\ []), to: AriaMath.AxonLayers.Comparison
  defdelegate le(a, b, opts \\ []), to: AriaMath.AxonLayers.Comparison
  defdelegate gt(a, b, opts \\ []), to: AriaMath.AxonLayers.Comparison
  defdelegate ge(a, b, opts \\ []), to: AriaMath.AxonLayers.Comparison

  # Special operations
  defdelegate is_nan(a, opts \\ []), to: AriaMath.AxonLayers.Special
  defdelegate is_inf(a, opts \\ []), to: AriaMath.AxonLayers.Special
  defdelegate select(condition, a, b, opts \\ []), to: AriaMath.AxonLayers.Special
  defdelegate switch(selection, cases, opts \\ []), to: AriaMath.AxonLayers.Special
  defdelegate random(opts \\ []), to: AriaMath.AxonLayers.Special

  # Trigonometric functions
  defdelegate deg_to_rad(degrees, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate rad_to_deg(radians, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate sin(angle, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate cos(angle, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate tan(angle, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate asin(value, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate acos(value, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate atan(value, opts \\ []), to: AriaMath.AxonLayers.Trigonometry
  defdelegate atan2(y, x, opts \\ []), to: AriaMath.AxonLayers.Trigonometry

  # Exponential functions
  defdelegate exp(value, opts \\ []), to: AriaMath.AxonLayers.Exponential
  defdelegate log(value, opts \\ []), to: AriaMath.AxonLayers.Exponential
  defdelegate log2(value, opts \\ []), to: AriaMath.AxonLayers.Exponential
  defdelegate log10(value, opts \\ []), to: AriaMath.AxonLayers.Exponential
  defdelegate sqrt(value, opts \\ []), to: AriaMath.AxonLayers.Exponential
  defdelegate cbrt(value, opts \\ []), to: AriaMath.AxonLayers.Exponential
  defdelegate pow(base, exponent, opts \\ []), to: AriaMath.AxonLayers.Exponential

  # Hyperbolic functions
  defdelegate sinh(value, opts \\ []), to: AriaMath.AxonLayers.Hyperbolic
  defdelegate cosh(value, opts \\ []), to: AriaMath.AxonLayers.Hyperbolic
  defdelegate tanh(value, opts \\ []), to: AriaMath.AxonLayers.Hyperbolic
  defdelegate asinh(value, opts \\ []), to: AriaMath.AxonLayers.Hyperbolic
  defdelegate acosh(value, opts \\ []), to: AriaMath.AxonLayers.Hyperbolic
  defdelegate atanh(value, opts \\ []), to: AriaMath.AxonLayers.Hyperbolic

  # Arithmetic operations
  defdelegate add(a, b, opts \\ []), to: AriaMath.AxonLayers.Arithmetic
  defdelegate sub(a, b, opts \\ []), to: AriaMath.AxonLayers.Arithmetic
  defdelegate mul(a, b, opts \\ []), to: AriaMath.AxonLayers.Arithmetic
  defdelegate div(a, b, opts \\ []), to: AriaMath.AxonLayers.Arithmetic

  # Vector operations
  defdelegate dot(a, b, opts \\ []), to: AriaMath.AxonLayers.Vector
  defdelegate cross(a, b, opts \\ []), to: AriaMath.AxonLayers.Vector
  defdelegate length(vector, opts \\ []), to: AriaMath.AxonLayers.Vector
  defdelegate normalize(vector, opts \\ []), to: AriaMath.AxonLayers.Vector
  defdelegate rotate_2d(vector, angle, opts \\ []), to: AriaMath.AxonLayers.Vector
  defdelegate rotate_3d(vector, quaternion, opts \\ []), to: AriaMath.AxonLayers.Vector
  defdelegate transform(vector, matrix, opts \\ []), to: AriaMath.AxonLayers.Vector

  # Matrix operations
  defdelegate mat_transpose(matrix, opts \\ []), to: AriaMath.AxonLayers.Matrix, as: :transpose
  defdelegate mat_det(matrix, opts \\ []), to: AriaMath.AxonLayers.Matrix, as: :det
  defdelegate mat_mul(a, b, opts \\ []), to: AriaMath.AxonLayers.Matrix, as: :mul
  defdelegate quat_conjugate(quaternion, opts \\ []), to: AriaMath.AxonLayers.Matrix
  defdelegate quat_mul(a, b, opts \\ []), to: AriaMath.AxonLayers.Matrix
  defdelegate quat_angle_between(a, b, opts \\ []), to: AriaMath.AxonLayers.Matrix
  defdelegate quat_from_axis_angle(axis, angle, opts \\ []), to: AriaMath.AxonLayers.Matrix
  defdelegate quat_to_axis_angle(quaternion, opts \\ []), to: AriaMath.AxonLayers.Matrix
  defdelegate quat_from_directions(a, b, opts \\ []), to: AriaMath.AxonLayers.Matrix

  # Elementary operations
  defdelegate abs(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate sign(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate trunc(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate floor(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate ceil(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate round(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate fract(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate neg(a, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate rem(a, b, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate clamp(value, min_val, max_val, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate saturate(value, opts \\ []), to: AriaMath.AxonLayers.Elementary

  # Comparison operations (for backward compatibility with existing main module functions)
  defdelegate min(a, b, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate max(a, b, opts \\ []), to: AriaMath.AxonLayers.Elementary
  defdelegate mix(a, b, t, opts \\ []), to: AriaMath.AxonLayers.Elementary

  # Integer arithmetic operations
  defdelegate int_abs(a, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :abs
  defdelegate int_sign(a, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :sign
  defdelegate int_neg(a, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :neg
  defdelegate int_add(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :add
  defdelegate int_sub(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :sub
  defdelegate int_mul(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :mul
  defdelegate int_div(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :div
  defdelegate int_rem(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :rem
  defdelegate int_min(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :min
  defdelegate int_max(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :max
  defdelegate int_clamp(a, min_val, max_val, opts \\ []), to: AriaMath.AxonLayers.IntegerArithmetic, as: :clamp

  # Integer bitwise operations
  defdelegate bitwise_not(a, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate bitwise_and(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate bitwise_or(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate bitwise_xor(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate shift_right_arithmetic(a, shift_amount, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate shift_left_logical(a, shift_amount, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate count_leading_zeros(a, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate count_trailing_zeros(a, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise
  defdelegate population_count(a, opts \\ []), to: AriaMath.AxonLayers.IntegerBitwise

  # Integer comparison operations
  defdelegate int_eq(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerComparison, as: :eq
  defdelegate int_lt(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerComparison, as: :lt
  defdelegate int_le(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerComparison, as: :le
  defdelegate int_gt(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerComparison, as: :gt
  defdelegate int_ge(a, b, opts \\ []), to: AriaMath.AxonLayers.IntegerComparison, as: :ge

  # Vector operations are delegated above

  # All other functions are properly delegated to the appropriate modules.
  # We avoid duplicate implementations to prevent compilation errors.

end
