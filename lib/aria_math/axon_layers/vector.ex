# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Vector do
  @moduledoc """
  Vector operations as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Computes the dot product of two vectors.
  Equivalent to math/dot from glTF interactivity specification.
  """
  def dot(a, b, opts \\ []) do
    name = opts[:name] || "dot"
    Axon.layer(&dot_impl/3, [a, b], name: name)
  end

  defnp dot_impl(a, b, _opts) do
    Nx.sum(Nx.multiply(a, b))
  end

  @doc """
  Computes the cross product of two 3D vectors.
  Equivalent to math/cross from glTF interactivity specification.
  """
  def cross(a, b, opts \\ []) do
    name = opts[:name] || "cross"
    Axon.layer(&cross_impl/3, [a, b], name: name)
  end

  defnp cross_impl(a, b, _opts) do
    # a, b: 3D vectors
    ax = a[0]
    ay = a[1]
    az = a[2]
    bx = b[0]
    by = b[1]
    bz = b[2]

    Nx.tensor([
      ay * bz - az * by,
      az * bx - ax * bz,
      ax * by - ay * bx
    ])
  end

  @doc """
  Length of vector.
  Equivalent to math/length from glTF interactivity specification.
  """
  def length(vector, opts \\ []) do
    name = opts[:name] || "length"
    Axon.layer(&length_impl/2, [vector], name: name)
  end

  defnp length_impl(vector, _opts) do
    # Euclidian length
    vector
    |> Nx.pow(2)
    |> Nx.sum()
    |> Nx.sqrt()
  end

  @doc """
  Normalize vector.
  Equivalent to math/normalize from glTF interactivity specification.
  """
  def normalize(vector, opts \\ []) do
    name = opts[:name] || "normalize"
    Axon.layer(&normalize_impl/2, [vector], name: name)
  end

  defnp normalize_impl(vector, _opts) do
    len = length_impl(vector, %{})
    Nx.divide(vector, len)
  end

  @doc """
  Performs 2D vector rotation by an angle.
  Equivalent to math/rotate2D from glTF interactivity specification.
  """
  def rotate_2d(vector, angle, opts \\ []) do
    name = opts[:name] || "rotate_2d"
    Axon.layer(&rotate_2d_impl/3, [vector, angle], name: name)
  end

  defnp rotate_2d_impl(vector, angle, _opts) do
    # vector: 2D vector, angle: scalar
    x = vector[0]
    y = vector[1]

    cos_a = Nx.cos(angle)
    sin_a = Nx.sin(angle)

    Nx.tensor([
      x * cos_a - y * sin_a,
      x * sin_a + y * cos_a
    ])
  end

  @doc """
  Performs 3D vector rotation using a quaternion.
  Equivalent to math/rotate3D from glTF interactivity specification.
  """
  def rotate_3d(vector, quaternion, opts \\ []) do
    name = opts[:name] || "rotate_3d"
    Axon.layer(&rotate_3d_impl/3, [vector, quaternion], name: name)
  end

  defnp rotate_3d_impl(vector, quaternion, _opts) do
    # vector: 3D vector, quaternion: [w, x, y, z] (unit quaternion)

    # Assumes quaternion is unit (as per glTF spec)
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    vx = vector[0]
    vy = vector[1]
    vz = vector[2]

    # Cross product: q × v
    rx = y * vz - z * vy
    ry = z * vx - x * vz
    rz = x * vy - y * vx

    # Cross product: q × (q × v)
    rrx = y * rz - z * ry
    rry = z * rx - x * rz
    rrz = x * ry - y * rx

    # Final rotation: v + 2 * ((q × v) * q_w + (q × (q × v)))
    Nx.tensor([
      vx + 2 * (rx * w + rrx),
      vy + 2 * (ry * w + rry),
      vz + 2 * (rz * w + rrz)
    ])
  end

  @doc """
  Transforms a vector with a transformation matrix.
  Equivalent to math/transform from glTF interactivity specification.
  """
  def transform(vector, matrix, opts \\ []) do
    name = opts[:name] || "transform"
    Axon.layer(&transform_impl/3, [vector, matrix], name: name)
  end

  defnp transform_impl(vector, matrix, _opts) do
    # Matrix multiplication
    Nx.dot(vector, matrix)
  end
end
