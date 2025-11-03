# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Comparison do
  @moduledoc """
  Comparison operations as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Equality comparison (component-wise).
  Equivalent to math/eq from glTF interactivity specification.

  If any input value is NaN, the output value is false.
  Negative zero equals positive zero.
  """
  def eq(a, b, opts \\ []) do
    name = opts[:name] || "eq"
    Axon.layer(&eq_impl/3, [a, b], name: name)
  end

  defnp eq_impl(a, b, _opts) do
    Nx.equal(a, b)
  end

  @doc """
  Less than comparison.
  Equivalent to math/lt from glTF interactivity specification.

  If any input value is NaN, the output value is false.
  """
  def lt(a, b, opts \\ []) do
    name = opts[:name] || "lt"
    Axon.layer(&lt_impl/3, [a, b], name: name)
  end

  defnp lt_impl(a, b, _opts) do
    Nx.less(a, b)
  end

  @doc """
  Less than or equal comparison.
  Equivalent to math/le from glTF interactivity specification.

  If any input value is NaN, the output value is false.
  """
  def le(a, b, opts \\ []) do
    name = opts[:name] || "le"
    Axon.layer(&le_impl/3, [a, b], name: name)
  end

  defnp le_impl(a, b, _opts) do
    Nx.less_equal(a, b)
  end

  @doc """
  Greater than comparison.
  Equivalent to math/gt from glTF interactivity specification.

  If any input value is NaN, the output value is false.
  """
  def gt(a, b, opts \\ []) do
    name = opts[:name] || "gt"
    Axon.layer(&gt_impl/3, [a, b], name: name)
  end

  defnp gt_impl(a, b, _opts) do
    Nx.greater(a, b)
  end

  @doc """
  Greater than or equal comparison.
  Equivalent to math/ge from glTF interactivity specification.

  If any input value is NaN, the output value is false.
  """
  def ge(a, b, opts \\ []) do
    name = opts[:name] || "ge"
    Axon.layer(&ge_impl/3, [a, b], name: name)
  end

  defnp ge_impl(a, b, _opts) do
    Nx.greater_equal(a, b)
  end
end
