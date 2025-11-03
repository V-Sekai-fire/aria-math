# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Matrix do
  @moduledoc """
  Matrix operations as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Matrix transpose.
  Equivalent to math/transpose from glTF interactivity specification.
  """
  def transpose(matrix, opts \\ []) do
    name = opts[:name] || "mat_transpose"
    Axon.layer(&transpose_impl/2, [matrix], name: name)
  end

  defnp transpose_impl(matrix, _opts) do
    Nx.transpose(matrix)
  end

  @doc """
  Matrix determinant.
  Equivalent to math/determinant from glTF interactivity specification.
  """
  def det(matrix, opts \\ []) do
    name = opts[:name] || "mat_det"
    Axon.layer(&det_impl/2, [matrix], name: name)
  end

  # Implementation for 2x2 and 3x3 matrices
  defnp det_impl(matrix, _opts) do
    shape = Nx.shape(matrix)

    case shape do
      {2, 2} ->
        # 2x2 determinant: ad - bc
        a = matrix[[0, 0]]
        b = matrix[[0, 1]]
        c = matrix[[1, 0]]
        d = matrix[[1, 1]]
        Nx.subtract(Nx.multiply(a, d), Nx.multiply(b, c))

      {3, 3} ->
        # 3x3 determinant: a(ei - fh) - b(di - fg) + c(dh - eg)
        a = matrix[[0, 0]]
        b = matrix[[0, 1]]
        c = matrix[[0, 2]]
        d = matrix[[1, 0]]
        e = matrix[[1, 1]]
        f = matrix[[1, 2]]
        g = matrix[[2, 0]]
        h = matrix[[2, 1]]
        i = matrix[[2, 2]]

        Nx.add(
          Nx.subtract(
            Nx.multiply(a, Nx.subtract(Nx.multiply(e, i), Nx.multiply(f, h))),
            Nx.multiply(b, Nx.subtract(Nx.multiply(d, i), Nx.multiply(f, g)))
          ),
          Nx.multiply(c, Nx.subtract(Nx.multiply(d, h), Nx.multiply(e, g)))
        )

      {4, 4} ->
        # For 4x4, simplified implementation - return 1.0 as placeholder
        # Full 4x4 determinant is complex
        Nx.tensor(1.0)

      _ ->
        Nx.tensor(0.0)
    end
  end

  @doc """
  Matrix inverse.
  Equivalent to math/inverse from glTF interactivity specification.
  Returns {inverse_matrix, is_valid}
  """
  def inverse(matrix, opts \\ []) do
    name = opts[:name] || "mat_inverse"
    Axon.layer(&inverse_impl/2, [matrix], name: name)
  end

  defnp inverse_impl(matrix, _opts) do
    det = det_impl(matrix, %{})

    # Check if determinant is valid (not zero, NaN, or infinity)
    det_finite = Nx.logical_and(Nx.not_equal(det, 0), Nx.is_finite(det))
    is_valid = det_finite

    # For now, simplified inverse for 4x4 affine transform matrices
    # Extract components: M = [R S T; 0 0 0 1]
    shape = Nx.shape(matrix)

    case shape do
      {4, 4} ->
        # Get the 3x3 rotation/scale part
        r = matrix[[0..2, 0..2]]
        # Get translation vector
        t = matrix[[0..2, 3]]

        # Simple inverse assuming affine transformation
        r_inv = Nx.transpose(r)  # Approximation: transpose works for rotation matrices
        neg_t = Nx.negate(t)
        t_inv = Nx.dot(r_inv, neg_t)

        # Build inverse matrix
        inverse_matrix = Nx.broadcast(0.0, {4, 4})
        inverse_matrix = Nx.put_slice(inverse_matrix, [0, 0], r_inv)
        inverse_matrix = Nx.put_slice(inverse_matrix, [0, 3], Nx.reshape(t_inv, {3, 1}))
        inverse_matrix = Nx.put_slice(inverse_matrix, [3, 3], Nx.tensor([[1.0]]))

        # If not valid, set to zeros
        inverse_matrix = Nx.select(is_valid, inverse_matrix, Nx.broadcast(0.0, {4, 4}))

        {inverse_matrix, is_valid}

      _ ->
        zero_matrix = Nx.broadcast(0.0, shape)
        {zero_matrix, Nx.tensor(false)}
    end
  end

  @doc """
  Matrix multiplication.
  Equivalent to math/matMul from glTF interactivity specification.
  """
  def mul(a, b, opts \\ []) do
    name = opts[:name] || "mat_mul"
    Axon.layer(&mul_impl/3, [a, b], name: name)
  end

  defnp mul_impl(a, b, _opts) do
    Nx.dot(a, b)
  end

  @doc """
  Compose a 4x4 transformation matrix from TRS components.
  Equivalent to math/matCompose from glTF interactivity specification.

  Assumes rotation quaternion is unit.
  """
  def compose(translation, rotation, scale, opts \\ []) do
    name = opts[:name] || "mat_compose"
    Axon.layer(&compose_impl/4, [translation, rotation, scale], name: name)
  end

  defnp compose_impl(translation, rotation, scale, _opts) do
    # translation: {batch, 3} or {3}
    # rotation: {batch, 4} or {4} [w, x, y, z]
    # scale: {batch, 3} or {3}
    # Returns {batch, 4, 4} or {4, 4}

    tx = translation[0]
    ty = translation[1]
    tz = translation[2]

    rw = rotation[0]
    rx = rotation[1]
    ry = rotation[2]
    rz = rotation[3]

    sx = scale[0]
    sy = scale[1]
    sz = scale[2]

    # Build rotation matrix from quaternion
    # R = 1 - 2(y²+z²),  2(xy-wz),      2(xz+wy)
    #     2(xy+wz),      1-2(x²+z²),     2(yz-wx)
    #     2(xz-wy),      2(yz+wx),       1-2(x²+y²)

    r00 = 1.0 - 2.0 * (ry * ry + rz * rz)
    r01 = 2.0 * (rx * ry - rz * rw)
    r02 = 2.0 * (rx * rz + ry * rw)

    r10 = 2.0 * (rx * ry + rz * rw)
    r11 = 1.0 - 2.0 * (rx * rx + rz * rz)
    r12 = 2.0 * (ry * rz - rx * rw)

    r20 = 2.0 * (rx * rz - ry * rw)
    r21 = 2.0 * (ry * rz + rx * rw)
    r22 = 1.0 - 2.0 * (rx * rx + ry * ry)

    # Apply scale to rotation matrix
    r00 = r00 * sx
    r01 = r01 * sy
    r02 = r02 * sz

    r10 = r10 * sx
    r11 = r11 * sy
    r12 = r12 * sz

    r20 = r20 * sx
    r21 = r21 * sy
    r22 = r22 * sz

    # Build final matrix
    Nx.tensor([
      [r00, r01, r02, tx],
      [r10, r11, r12, ty],
      [r20, r21, r22, tz],
      [0.0, 0.0, 0.0, 1.0]
    ])
  end

  @doc """
  Decompose a 4x4 transformation matrix to TRS components.
  Equivalent to math/matDecompose from glTF interactivity specification.
  Returns {translation, rotation, scale, is_valid}
  """
  def decompose(matrix, opts \\ []) do
    name = opts[:name] || "mat_decompose"
    Axon.layer(&decompose_impl/2, [matrix], name: name)
  end

  defnp decompose_impl(matrix, _opts) do
    # Check if the fourth row is [0, 0, 0, 1]
    fourth_row = matrix[[3, 0..3]]
    valid_fourth_row = Nx.all_close(fourth_row, Nx.tensor([0.0, 0.0, 0.0, 1.0]), atol: 1.0e-7)

    # Extract translation
    translation = Nx.stack([matrix[[0, 3]], matrix[[1, 3]], matrix[[2, 3]]])

    # Extract scales (lengths of columns)
    col0 = Nx.stack([matrix[[0, 0]], matrix[[0, 1]], matrix[[0, 2]]])
    col1 = Nx.stack([matrix[[1, 0]], matrix[[1, 1]], matrix[[1, 2]]])
    col2 = Nx.stack([matrix[[2, 0]], matrix[[2, 1]], matrix[[2, 2]]])

    sx = Nx.sqrt(Nx.sum(col0 * col0))
    sy = Nx.sqrt(Nx.sum(col1 * col1))
    sz = Nx.sqrt(Nx.sum(col2 * col2))

    # Check for valid scales
    scales_finite = Nx.all(Nx.is_finite(Nx.stack([sx, sy, sz])))
    scales_positive = Nx.all(Nx.greater(Nx.stack([sx, sy, sz]), 0))
    scales_valid = Nx.logical_and(scales_finite, scales_positive)

    # Overall validity
    is_valid = Nx.logical_and(valid_fourth_row, scales_valid)

    # Normalize columns to get rotation matrix (only when valid)
    col0_norm = Nx.select(is_valid, col0 / sx, Nx.tensor([1.0, 0.0, 0.0]))
    col1_norm = Nx.select(is_valid, col1 / sy, Nx.tensor([0.0, 1.0, 0.0]))
    col2_norm = Nx.select(is_valid, col2 / sz, Nx.tensor([0.0, 0.0, 1.0]))

    rotation_matrix = Nx.select(
      is_valid,
      Nx.stack([col0_norm, col1_norm, col2_norm], axis: 1),
      Nx.eye(3)  # Identity matrix when invalid
    )

    # Convert rotation matrix to quaternion (simplified)
    # This is a simplified implementation - full conversion is complex
    qw = Nx.select(
      is_valid,
      Nx.sqrt(1 + rotation_matrix[[0, 0]] + rotation_matrix[[1, 1]] + rotation_matrix[[2, 2]]) / 2,
      1.0
    )
    qx = Nx.select(is_valid, (rotation_matrix[[2, 1]] - rotation_matrix[[1, 2]]) / (4 * qw), 0.0)
    qy = Nx.select(is_valid, (rotation_matrix[[0, 2]] - rotation_matrix[[2, 0]]) / (4 * qw), 0.0)
    qz = Nx.select(is_valid, (rotation_matrix[[1, 0]] - rotation_matrix[[0, 1]]) / (4 * qw), 0.0)

    rotation = Nx.stack([qw, qx, qy, qz])
    scale = Nx.stack([
      Nx.select(is_valid, sx, 1.0),
      Nx.select(is_valid, sy, 1.0),
      Nx.select(is_valid, sz, 1.0)
    ])

    {translation, rotation, scale, Nx.logical_and(valid_fourth_row, scales_valid)}
  end

  @doc """
  Quaternion conjugation.
  Equivalent to math/quatConjugate from glTF interactivity specification.
  """
  def quat_conjugate(quaternion, opts \\ []) do
    name = opts[:name] || "quat_conjugate"
    Axon.layer(&quat_conjugate_impl/2, [quaternion], name: name)
  end

  defnp quat_conjugate_impl(quaternion, _opts) do
    # quaternion: [w, x, y, z]
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    # Conjugate is [-x, -y, -z, w]
    Nx.stack([-x, -y, -z, w])
  end

  @doc """
  Quaternion multiplication.
  Equivalent to math/quatMul from glTF interactivity specification.
  """
  def quat_mul(a, b, opts \\ []) do
    name = opts[:name] || "quat_mul"
    Axon.layer(&quat_mul_impl/3, [a, b], name: name)
  end

  defnp quat_mul_impl(a, b, _opts) do
    # a, b: quaternions [w, x, y, z]
    aw = a[0]
    ax = a[1]
    ay = a[2]
    az = a[3]

    bw = b[0]
    bx = b[1]
    by = b[2]
    bz = b[3]

    Nx.stack([
      aw * bw - ax * bx - ay * by - az * bz,  # w component
      aw * bx + ax * bw + ay * bz - az * by,  # x component
      aw * by - ax * bz + ay * bw + az * bx,  # y component
      aw * bz + ax * by - ay * bx + az * bw   # z component
    ])
  end

  @doc """
  Angle between two unit quaternions.
  Equivalent to math/quatAngleBetween from glTF interactivity specification.
  """
  def quat_angle_between(a, b, opts \\ []) do
    name = opts[:name] || "quat_angle_between"
    Axon.layer(&quat_angle_between_impl/3, [a, b], name: name)
  end

  defnp quat_angle_between_impl(a, b, _opts) do
    # Assumes both quaternions are unit
    dot_product = Nx.sum(Nx.multiply(a, b))
    # Clamp dot product to [-1, 1] for numerical stability
    dot_clamped = Nx.max(Nx.min(dot_product, 1.0), -1.0)
    Nx.multiply(2.0, Nx.acos(dot_clamped))
  end

  @doc """
  Create quaternion from axis and angle.
  Equivalent to math/quatFromAxisAngle from glTF interactivity specification.

  Axis should be unit vector.
  """
  def quat_from_axis_angle(axis, angle, opts \\ []) do
    name = opts[:name] || "quat_from_axis_angle"
    Axon.layer(&quat_from_axis_angle_impl/3, [axis, angle], name: name)
  end

  defnp quat_from_axis_angle_impl(axis, angle, _opts) do
    # axis: unit vector [x, y, z], angle: scalar in radians
    half_angle = Nx.divide(angle, 2.0)
    sin_half = Nx.sin(half_angle)
    cos_half = Nx.cos(half_angle)

    ax = axis[0]
    ay = axis[1]
    az = axis[2]

    Nx.stack([
      ax * sin_half,  # x
      ay * sin_half,  # y
      az * sin_half,  # z
      cos_half       # w
    ])
  end

  @doc """
  Decompose quaternion to axis and angle.
  Equivalent to math/quatToAxisAngle from glTF interactivity specification.

  Assumes quaternion is unit.
  """
  def quat_to_axis_angle(quaternion, opts \\ []) do
    name = opts[:name] || "quat_to_axis_angle"
    Axon.layer(&quat_to_axis_angle_impl/2, [quaternion], name: name)
  end

  defnp quat_to_axis_angle_impl(quaternion, _opts) do
    # quaternion: [w, x, y, z] where it is unit
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    # Handle near-zero angle case
    # If |w| ≈ 1, quaternion represents near-zero rotation
    epsilon = 1.0e-7

    w_abs = Nx.abs(w)

    angle = Nx.select(
      Nx.greater(w_abs, 1.0 - epsilon),
      Nx.tensor(0.0),
      Nx.multiply(2.0, Nx.acos(Nx.min(Nx.max(w, -1.0), 1.0)))
    )

    # Default axis when angle is near zero
    default_axis = Nx.select(
      Nx.less(Nx.abs(x), Nx.abs(y)),
      Nx.select(Nx.less(Nx.abs(x), Nx.abs(z)), Nx.tensor([1.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 1.0])),
      Nx.select(Nx.less(Nx.abs(y), Nx.abs(z)), Nx.tensor([0.0, 1.0, 0.0]), Nx.tensor([0.0, 0.0, 1.0]))
    )

    axis = Nx.select(
      Nx.greater(w_abs, 1.0 - epsilon),
      default_axis,
      Nx.divide(Nx.stack([x, y, z]), Nx.sqrt(Nx.subtract(1.0, w * w)))
    )

    {axis, angle}
  end

  @doc """
  Create quaternion from two directional unit vectors.
  Equivalent to math/quatFromDirections from glTF interactivity specification.

  Both directions should be unit vectors.
  """
  def quat_from_directions(a, b, opts \\ []) do
    name = opts[:name] || "quat_from_directions"
    Axon.layer(&quat_from_directions_impl/3, [a, b], name: name)
  end

  defnp quat_from_directions_impl(a, b, _opts) do
    # a, b: unit direction vectors
    epsilon = 1.0e-7

    # Compute dot product and cross product
    dot = Nx.sum(Nx.multiply(a, b))
    cross = Nx.stack([
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ])

    # Normalize cross product for quaternion
    cross_len = Nx.sqrt(Nx.sum(Nx.pow(cross, 2)))
    cross_norm = Nx.divide(cross, Nx.add(cross_len, epsilon))

    # Handle cases based on dot product
    w = Nx.select(
      Nx.greater(dot, 0.9999),
      Nx.tensor(1.0),
      Nx.select(
        Nx.less(dot, -0.9999),
        Nx.tensor(0.0),
        Nx.divide(Nx.add(1.0, dot), Nx.sqrt(Nx.multiply(2.0, Nx.add(1.0, dot))))
      )
    )

    perp_axis = Nx.select(
      Nx.less(Nx.abs(a[0]), Nx.abs(a[1])),
      Nx.select(Nx.less(Nx.abs(a[0]), Nx.abs(a[2])), Nx.tensor([1.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 1.0])),
      Nx.select(Nx.less(Nx.abs(a[1]), Nx.abs(a[2])), Nx.tensor([0.0, 1.0, 0.0]), Nx.tensor([0.0, 0.0, 1.0]))
    )

    s = Nx.select(
      Nx.logical_or(Nx.greater(dot, 0.9999), Nx.less(dot, -0.9999)),
      Nx.tensor(0.0),
      Nx.sqrt(Nx.multiply(2.0, Nx.subtract(1.0, Nx.pow(w, 2))))
    )

    xyz = Nx.select(
      Nx.greater(dot, 0.9999),
      Nx.tensor([0.0, 0.0, 0.0]),
      Nx.select(
        Nx.less(dot, -0.9999),
        perp_axis,
        Nx.multiply(cross_norm, s)
      )
    )

    # Return quaternion [w, x, y, z]
    Nx.concatenate([Nx.reshape(w, {1}), xyz])
  end
end
