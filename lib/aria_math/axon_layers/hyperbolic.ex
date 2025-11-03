# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Hyperbolic do
  @moduledoc """
  Hyperbolic and inverse hyperbolic functions as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Hyperbolic sine (component-wise).
  Equivalent to math/sinh from glTF interactivity specification.
  """
  def sinh(value, opts \\ []) do
    name = opts[:name] || "sinh"
    Axon.layer(&sinh_impl/2, [value], name: name)
  end

  defnp sinh_impl(value, _opts) do
    Nx.sinh(value)
  end

  @doc """
  Hyperbolic cosine (component-wise).
  Equivalent to math/cosh from glTF interactivity specification.
  """
  def cosh(value, opts \\ []) do
    name = opts[:name] || "cosh"
    Axon.layer(&cosh_impl/2, [value], name: name)
  end

  defnp cosh_impl(value, _opts) do
    Nx.cosh(value)
  end

  @doc """
  Hyperbolic tangent (component-wise).
  Equivalent to math/tanh from glTF interactivity specification.
  """
  def tanh(value, opts \\ []) do
    name = opts[:name] || "tanh"
    Axon.layer(&tanh_impl/2, [value], name: name)
  end

  defnp tanh_impl(value, _opts) do
    Nx.tanh(value)
  end

  @doc """
  Inverse hyperbolic sine (component-wise).
  Equivalent to math/asinh from glTF interactivity specification.
  """
  def asinh(value, opts \\ []) do
    name = opts[:name] || "asinh"
    Axon.layer(&asinh_impl/2, [value], name: name)
  end

  defnp asinh_impl(value, _opts) do
    Nx.asinh(value)
  end

  @doc """
  Inverse hyperbolic cosine (component-wise).
  Equivalent to math/acosh from glTF interactivity specification.
  """
  def acosh(value, opts \\ []) do
    name = opts[:name] || "acosh"
    Axon.layer(&acosh_impl/2, [value], name: name)
  end

  defnp acosh_impl(value, _opts) do
    Nx.acosh(value)
  end

  @doc """
  Inverse hyperbolic tangent (component-wise).
  Equivalent to math/atanh from glTF interactivity specification.
  """
  def atanh(value, opts \\ []) do
    name = opts[:name] || "atanh"
    Axon.layer(&atanh_impl/2, [value], name: name)
  end

  defnp atanh_impl(value, _opts) do
    Nx.atanh(value)
  end
end
