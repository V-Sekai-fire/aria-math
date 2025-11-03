# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.Vector3 do
  @moduledoc """
  Differentiable 3D Vector mathematics using Axon layers for gradient-based learning.

  Provides vector operations that can participate in neural network training and
  optimization, enabling learned geometric transformations and spatial reasoning.
  """

  import Nx.Defn
  alias Axon.Tensor, as: A

  @doc """
  Calculate the magnitude (length) of a 3D vector as an Axon layer.

  Supports gradients for length-based optimization in geometric learning.
  """
  @spec magnitude(A.t(), keyword()) :: A.t()
  def magnitude(vector, opts \\ []) do
    name = opts[:name] || "vector_magnitude"
    Axon.layer(&magnitude_impl/2, [vector], name: name)
  end

  defnp magnitude_impl(vector, _opts) do
    Nx.sqrt(Nx.sum(Nx.pow(vector, 2)))
  end

  @doc """
  Normalize a 3D vector to unit length as an Axon layer.

  Learnable vector normalization for neural geometric processing.
  """
  @spec normalize(A.t(), keyword()) :: A.t()
  def normalize(vector, opts \\ []) do
    name = opts[:name] || "vector_normalize"
    Axon.layer(&normalize_impl/2, [vector], name: name)
  end

  defnp normalize_impl(vector, _opts) do
    mag = magnitude_impl(vector, %{})
    # Avoid division by zero
    safe_mag = Nx.select(Nx.equal(mag, 0.0), 1.0, mag)
    Nx.divide(vector, safe_mag)
  end

  @doc """
  Calculate the dot product of two 3D vectors as an Axon layer.

  Supports learned similarity metrics and geometric relationships.
  """
  @spec dot(A.t(), A.t(), keyword()) :: A.t()
  def dot(a, b, opts \\ []) do
    name = opts[:name] || "vector_dot"
    Axon.layer(&dot_impl/3, [a, b], name: name)
  end

  defnp dot_impl(a, b, _opts) do
    Nx.sum(Nx.multiply(a, b))
  end

  @doc """
  Calculate the cross product of two 3D vectors as an Axon layer.

  Enables learned 3D spatial transformations and coordinate system operations.
  """
  @spec cross(A.t(), A.t(), keyword()) :: A.t()
  def cross(a, b, opts \\ []) do
    name = opts[:name] || "vector_cross"
    Axon.layer(&cross_impl/3, [a, b], name: name)
  end

  defnp cross_impl(a, b, _opts) do
    # Standard cross product: a × b
    ax = a[0]
    ay = a[1]
    az = a[2]
    bx = b[0]
    by = b[1]
    bz = b[2]

    Nx.stack([
      ay * bz - az * by,
      az * bx - ax * bz,
      ax * by - ay * bx
    ])
  end

  @doc """
  Add two 3D vectors as an Axon layer.

  Supports learned vector composition and accumulation operations.
  """
  @spec add(A.t(), A.t(), keyword()) :: A.t()
  def add(a, b, opts \\ []) do
    name = opts[:name] || "vector_add"
    Axon.layer(&add_impl/3, [a, b], name: name)
  end

  defnp add_impl(a, b, _opts) do
    Nx.add(a, b)
  end

  @doc """
  Subtract two 3D vectors as an Axon layer.

  Enables learned vector differences and relative positioning.
  """
  @spec subtract(A.t(), A.t(), keyword()) :: A.t()
  def subtract(a, b, opts \\ []) do
    name = opts[:name] || "vector_subtract"
    Axon.layer(&subtract_impl/3, [a, b], name: name)
  end

  defnp subtract_impl(a, b, _opts) do
    Nx.subtract(a, b)
  end

  @doc """
  Scale a 3D vector by a learnable scalar as an Axon layer.

  Supports scaling optimization in geometric deep learning.
  """
  @spec scale(A.t(), A.t(), keyword()) :: A.t()
  def scale(vector, scalar, opts \\ []) do
    name = opts[:name] || "vector_scale"
    Axon.layer(&scale_impl/3, [vector, scalar], name: name)
  end

  defnp scale_impl(vector, scalar, _opts) do
    Nx.multiply(vector, scalar)
  end

  @doc """
  Calculate the Euclidean distance between two 3D vectors as an Axon layer.

  Useful for learned distance metrics and spatial proximity optimization.
  """
  @spec distance(A.t(), A.t(), keyword()) :: A.t()
  def distance(a, b, opts \\ []) do
    name = opts[:name] || "vector_distance"
    Axon.layer(&distance_impl/3, [a, b], name: name)
  end

  defnp distance_impl(a, b, _opts) do
    diff = Nx.subtract(a, b)
    magnitude_impl(diff, %{})
  end

  @doc """
  Linear interpolation between two 3D vectors as an Axon layer.

  Enables learned blending and morphing operations in geometric spaces.
  """
  @spec lerp(A.t(), A.t(), A.t(), keyword()) :: A.t()
  def lerp(a, b, t, opts \\ []) do
    name = opts[:name] || "vector_lerp"
    Axon.layer(&lerp_impl/4, [a, b, t], name: name)
  end

  defnp lerp_impl(a, b, t, _opts) do
    Nx.add(a, Nx.multiply(Nx.subtract(b, a), t))
  end

  @doc """
  Project vector A onto vector B as an Axon layer.

  Supports learned geometric projections and decomposition.
  """
  @spec project(A.t(), A.t(), keyword()) :: A.t()
  def project(a, b, opts \\ []) do
    name = opts[:name] || "vector_project"
    Axon.layer(&project_impl/3, [a, b], name: name)
  end

  defnp project_impl(a, b, _opts) do
    # proj_b(a) = (a · b) / (b · b) * b
    dot_ab = dot_impl(a, b, %{})
    dot_bb = dot_impl(b, b, %{})
    safe_dot_bb = Nx.select(Nx.equal(dot_bb, 0.0), 1.0, dot_bb)
    scale_factor = Nx.divide(dot_ab, safe_dot_bb)
    Nx.multiply(b, scale_factor)
  end

  @doc """
  Reflect vector A across normal vector B as an Axon layer.

  Enables learned reflections in geometric optimization.
  """
  @spec reflect(A.t(), A.t(), keyword()) :: A.t()
  def reflect(a, normal, opts \\ []) do
    name = opts[:name] || "vector_reflect"
    Axon.layer(&reflect_impl/3, [a, normal], name: name)
  end

  defnp reflect_impl(a, normal, _opts) do
    # reflect(a, n) = a - 2(a · n)n
    # Assuming normal is normalized
    dot_an = dot_impl(a, normal, %{})
    two_dot = Nx.multiply(dot_an, 2.0)
    scaled_normal = Nx.multiply(normal, two_dot)
    Nx.subtract(a, scaled_normal)
  end
end
