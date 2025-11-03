# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Special do
  @moduledoc """
  Special mathematical operations as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Is Not a Number check (component-wise).
  Equivalent to math/isNaN from glTF interactivity specification.
  """
  def is_nan(a, opts \\ []) do
    name = opts[:name] || "is_nan"
    Axon.layer(&is_nan_impl/2, [a], name: name)
  end

  defnp is_nan_impl(a, _opts) do
    Nx.is_nan(a)
  end

  @doc """
  Is Infinity check (component-wise).
  Equivalent to math/isInf from glTF interactivity specification.
  """
  def is_inf(a, opts \\ []) do
    name = opts[:name] || "is_inf"
    Axon.layer(&is_inf_impl/2, [a], name: name)
  end

  defnp is_inf_impl(a, _opts) do
    Nx.is_infinite(a)
  end

  @doc """
  Conditional selection.
  Equivalent to math/select from glTF interactivity specification.

  The type T represents any supported type including custom types.
  It MUST be the same for the output and both input options.
  """
  def select(condition, a, b, opts \\ []) do
    name = opts[:name] || "select"
    Axon.layer(&select_impl/4, [condition, a, b], name: name)
  end

  defnp select_impl(condition, a, b, _opts) do
    Nx.select(condition, a, b)
  end

  @doc """
  Multi-case conditional switching.
  Equivalent to math/switch from glTF interactivity specification.

  Note: simplified implementation - full switch with dynamic cases is complex in Axon.
  This is a basic version that switches between 0 and 1.
  """
  def switch(selection, cases, opts \\ []) do
    name = opts[:name] || "switch"
    Axon.layer(&switch_impl/3, [selection, cases], name: name)
  end

  defnp switch_impl(selection, cases, _opts) do
    # Simplified switch implementation
    # selection: integer index
    # cases: list of values to choose from
    Nx.select(Nx.equal(selection, 0), cases[0], cases[1])
  end

  @doc """
  Random number generation (pseudo-random).
  Equivalent to math/random from glTF interactivity specification.

  Returns a pseudo-random number greater than or equal to zero and less than one.
  The value is cached and only updated when accessed as a result of a new computation.
  """
  def random(opts \\ []) do
    name = opts[:name] || "random"
    Axon.layer(&random_impl/1, [], name: name)
  end

  defnp random_impl(_opts) do
    Nx.random_uniform(shape: {}, min: 0.0, max: 1.0)
  end
end
