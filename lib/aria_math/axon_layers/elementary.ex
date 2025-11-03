# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Elementary do
  @moduledoc """
  Elementary mathematical functions as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Absolute value (component-wise).
  Equivalent to math/abs from glTF interactivity specification.
  """
  def abs(a, opts \\ []) do
    name = opts[:name] || "abs"
    Axon.layer(&abs_impl/2, [a], name: name)
  end

  defnp abs_impl(a, _opts) do
    Nx.abs(a)
  end

  @doc """
  Sign operation (component-wise).
  Equivalent to math/sign from glTF spec.
  """
  def sign(a, opts \\ []) do
    name = opts[:name] || "sign"
    Axon.layer(&sign_impl/2, [a], name: name)
  end

  defnp sign_impl(a, _opts) do
    Nx.sign(a)
  end

  @doc """
  Truncate operation (component-wise).
  Equivalent to math/trunc from glTF spec.
  """
  def trunc(a, opts \\ []) do
    name = opts[:name] || "trunc"
    Axon.layer(&trunc_impl/2, [a], name: name)
  end

  defnp trunc_impl(a, _opts) do
    Nx.select(Nx.greater_equal(a, 0), Nx.floor(a), Nx.ceil(a))
  end

  @doc """
  Floor operation (component-wise).
  Equivalent to math/floor from glTF spec.
  """
  def floor(a, opts \\ []) do
    name = opts[:name] || "floor"
    Axon.layer(&floor_impl/2, [a], name: name)
  end

  defnp floor_impl(a, _opts) do
    Nx.floor(a)
  end

  @doc """
  Ceiling operation (component-wise).
  Equivalent to math/ceil from glTF spec.
  """
  def ceil(a, opts \\ []) do
    name = opts[:name] || "ceil"
    Axon.layer(&ceil_impl/2, [a], name: name)
  end

  defnp ceil_impl(a, _opts) do
    Nx.ceil(a)
  end

  @doc """
  Round operation (component-wise).
  Equivalent to math/round from glTF spec.
  """
  def round(a, opts \\ []) do
    name = opts[:name] || "round"
    Axon.layer(&round_impl/2, [a], name: name)
  end

  defnp round_impl(a, _opts) do
    Nx.round(a)
  end

  @doc """
  Fractional part (component-wise).
  Equivalent to math/fract from glTF spec.
  """
  def fract(a, opts \\ []) do
    name = opts[:name] || "fract"
    Axon.layer(&fract_impl/2, [a], name: name)
  end

  defnp fract_impl(a, _opts) do
    # a - floor(a)
    Nx.subtract(a, Nx.floor(a))
  end

  @doc """
  Negation operation (component-wise).
  Equivalent to math/neg from glTF spec.
  """
  def neg(a, opts \\ []) do
    name = opts[:name] || "neg"
    Axon.layer(&neg_impl/2, [a], name: name)
  end

  defnp neg_impl(a, _opts) do
    Nx.negate(a)
  end

  @doc """
  Remainder operation (component-wise).
  Equivalent to math/rem from glTF spec.
  """
  def rem(a, b, opts \\ []) do
    name = opts[:name] || "rem"
    Axon.layer(&rem_impl/3, [a, b], name: name)
  end

  defnp rem_impl(a, b, _opts) do
    Nx.remainder(a, b)
  end

  @doc """
  Clamping operation (component-wise).
  Equivalent to math/clamp from glTF spec.
  """
  def clamp(value, min_val, max_val, opts \\ []) do
    name = opts[:name] || "clamp"
    Axon.layer(&clamp_impl/4, [value, min_val, max_val], name: name)
  end

  defnp clamp_impl(value, min_val, max_val, _opts) do
    Nx.max(Nx.min(value, max_val), min_val)
  end

  @doc """
  Saturation (clamping to [0, 1]) (component-wise).
  Equivalent to math/saturate from glTF spec.
  """
  def saturate(value, opts \\ []) do
    name = opts[:name] || "saturate"
    Axon.layer(&saturate_impl/2, [value], name: name)
  end

  defnp saturate_impl(value, _opts) do
    Nx.max(Nx.min(value, 1), 0)
  end
end
