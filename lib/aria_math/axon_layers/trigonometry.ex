# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Trigonometry do
  @moduledoc """
  Trigonometric and inverse trigonometric functions as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Degrees to radians.
  Equivalent to math/rad from glTF interactivity specification.
  """
  def deg_to_rad(degrees, opts \\ []) do
    name = opts[:name] || "deg_to_rad"
    Axon.layer(&deg_to_rad_impl/2, [degrees], name: name)
  end

  defnp deg_to_rad_impl(degrees, _opts) do
    Nx.divide(Nx.multiply(degrees, Nx.PI), 180)
  end

  @doc """
  Radians to degrees.
  Equivalent to math/deg from glTF interactivity specification.
  """
  def rad_to_deg(radians, opts \\ []) do
    name = opts[:name] || "rad_to_deg"
    Axon.layer(&rad_to_deg_impl/2, [radians], name: name)
  end

  defnp rad_to_deg_impl(radians, _opts) do
    Nx.divide(Nx.multiply(radians, 180), Nx.PI)
  end

  @doc """
  Sine (component-wise).
  Equivalent to math/sin from glTF interactivity specification.
  """
  def sin(angle, opts \\ []) do
    name = opts[:name] || "sin"
    Axon.layer(&sin_impl/2, [angle], name: name)
  end

  defnp sin_impl(angle, _opts) do
    Nx.sin(angle)
  end

  @doc """
  Cosine (component-wise).
  Equivalent to math/cos from glTF interactivity specification.
  """
  def cos(angle, opts \\ []) do
    name = opts[:name] || "cos"
    Axon.layer(&cos_impl/2, [angle], name: name)
  end

  defnp cos_impl(angle, _opts) do
    Nx.cos(angle)
  end

  @doc """
  Tangent (component-wise).
  Equivalent to math/tan from glTF interactivity specification.
  """
  def tan(angle, opts \\ []) do
    name = opts[:name] || "tan"
    Axon.layer(&tan_impl/2, [angle], name: name)
  end

  defnp tan_impl(angle, _opts) do
    Nx.tan(angle)
  end

  @doc """
  Arcsine (component-wise).
  Equivalent to math/asin from glTF interactivity specification.
  """
  def asin(value, opts \\ []) do
    name = opts[:name] || "asin"
    Axon.layer(&asin_impl/2, [value], name: name)
  end

  defnp asin_impl(value, _opts) do
    Nx.asin(value)
  end

  @doc """
  Arccosine (component-wise).
  Equivalent to math/acos from glTF interactivity specification.
  """
  def acos(value, opts \\ []) do
    name = opts[:name] || "acos"
    Axon.layer(&acos_impl/2, [value], name: name)
  end

  defnp acos_impl(value, _opts) do
    Nx.acos(value)
  end

  @doc """
  Arctangent (component-wise).
  Equivalent to math/atan from glTF interactivity specification.
  """
  def atan(value, opts \\ []) do
    name = opts[:name] || "atan"
    Axon.layer(&atan_impl/2, [value], name: name)
  end

  defnp atan_impl(value, _opts) do
    Nx.atan(value)
  end

  @doc """
  Arctangent of y/x (component-wise).
  Equivalent to math/atan2 from glTF interactivity specification.
  """
  def atan2(y, x, opts \\ []) do
    name = opts[:name] || "atan2"
    Axon.layer(&atan2_impl/3, [y, x], name: name)
  end

  defnp atan2_impl(y, x, _opts) do
    Nx.atan2(y, x)
  end
end
