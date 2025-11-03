# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Exponential do
  @moduledoc """
  Exponential and logarithmic functions as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Exponential (component-wise).
  Equivalent to math/exp from glTF interactivity specification.
  """
  def exp(value, opts \\ []) do
    name = opts[:name] || "exp"
    Axon.layer(&exp_impl/2, [value], name: name)
  end

  defnp exp_impl(value, _opts) do
    Nx.exp(value)
  end

  @doc """
  Natural logarithm (component-wise).
  Equivalent to math/log from glTF interactivity specification.
  """
  def log(value, opts \\ []) do
    name = opts[:name] || "log"
    Axon.layer(&log_impl/2, [value], name: name)
  end

  defnp log_impl(value, _opts) do
    Nx.log(value)
  end

  @doc """
  Base-2 logarithm (component-wise).
  Equivalent to math/log2 from glTF interactivity specification.
  """
  def log2(value, opts \\ []) do
    name = opts[:name] || "log2"
    Axon.layer(&log2_impl/2, [value], name: name)
  end

  defnp log2_impl(value, _opts) do
    Nx.divide(Nx.log(value), Nx.log(2))
  end

  @doc """
  Base-10 logarithm (component-wise).
  Equivalent to math/log10 from glTF interactivity specification.
  """
  def log10(value, opts \\ []) do
    name = opts[:name] || "log10"
    Axon.layer(&log10_impl/2, [value], name: name)
  end

  defnp log10_impl(value, _opts) do
    Nx.divide(Nx.log(value), Nx.log(10))
  end

  @doc """
  Square root (component-wise).
  Equivalent to math/sqrt from glTF interactivity specification.
  """
  def sqrt(value, opts \\ []) do
    name = opts[:name] || "sqrt"
    Axon.layer(&sqrt_impl/2, [value], name: name)
  end

  defnp sqrt_impl(value, _opts) do
    Nx.sqrt(value)
  end

  @doc """
  Cube root (component-wise).
  Equivalent to math/cbrt from glTF interactivity specification.
  """
  def cbrt(value, opts \\ []) do
    name = opts[:name] || "cbrt"
    Axon.layer(&cbrt_impl/2, [value], name: name)
  end

  defnp cbrt_impl(value, _opts) do
    Nx.pow(value, 1/3)
  end

  @doc """
  Power function (component-wise).
  Equivalent to math/pow from glTF interactivity specification.
  """
  def pow(base, exponent, opts \\ []) do
    name = opts[:name] || "pow"
    Axon.layer(&pow_impl/3, [base, exponent], name: name)
  end

  defnp pow_impl(base, exponent, _opts) do
    Nx.pow(base, exponent)
  end
end
