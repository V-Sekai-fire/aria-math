# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.IntegerArithmetic do
  @moduledoc """
  Integer arithmetic operations as Axon layers.
  Compatible with glTF interactivity specification.
  All inputs/outputs are two's complement 32-bit signed integers.
  Arithmetic overflow wraps around (using `(value | 0)` behavior).
  """

  import Nx.Defn

  @doc """
  Integer absolute value.
  Equivalent to math/abs from glTF integer arithmetic specification.
  Note: abs(-2147483648) = -2147483648 (overflow behavior).
  """
  def abs(a, opts \\ []) do
    name = opts[:name] || "int_abs"
    Axon.layer(&abs_impl/2, [a], name: name)
  end

  defnp abs_impl(a, _opts) do
    # This behaves correctly for integer overflow case
    Nx.abs(a) |> Nx.as_type({:s, 32})
  end

  @doc """
  Integer sign operation.
  Returns -1 for negative, 0 for zero, 1 for positive.
  Equivalent to math/sign from glTF integer arithmetic specification.
  """
  def sign(a, opts \\ []) do
    name = opts[:name] || "int_sign"
    Axon.layer(&sign_impl/2, [a], name: name)
  end

  defnp sign_impl(a, _opts) do
    Nx.sign(a) |> Nx.as_type({:s, 32})
  end

  @doc """
  Integer negation.
  Equivalent to math/neg from glTF integer arithmetic specification.
  Note: neg(-2147483648) = -2147483648 (overflow behavior).
  """
  def neg(a, opts \\ []) do
    name = opts[:name] || "int_neg"
    Axon.layer(&neg_impl/2, [a], name: name)
  end

  defnp neg_impl(a, _opts) do
    Nx.negate(a) |> Nx.as_type({:s, 32})
  end

  @doc """
  Integer addition.
  Equivalent to math/add from glTF integer arithmetic specification.
  Overflow wraps around.
  """
  def add(a, b, opts \\ []) do
    name = opts[:name] || "int_add"
    Axon.layer(&add_impl/3, [a, b], name: name)
  end

  defnp add_impl(a, b, _opts) do
    Nx.add(a, b) |> Nx.as_type({:s, 32})
  end

  @doc """
  Integer subtraction.
  Equivalent to math/sub from glTF integer arithmetic specification.
  Overflow wraps around.
  """
  def sub(a, b, opts \\ []) do
    name = opts[:name] || "int_sub"
    Axon.layer(&sub_impl/3, [a, b], name: name)
  end

  defnp sub_impl(a, b, _opts) do
    Nx.subtract(a, b) |> Nx.as_type({:s, 32})
  end

  @doc """
  Integer multiplication.
  Equivalent to math/mul from glTF integer arithmetic specification.
  Overflow wraps around.
  """
  def mul(a, b, opts \\ []) do
    name = opts[:name] || "int_mul"
    Axon.layer(&mul_impl/3, [a, b], name: name)
  end

  defnp mul_impl(a, b, _opts) do
    # Math.imul behavior - use 64-bit intermediate, then truncate to 32-bit
    result = Nx.multiply(a, b)
    Nx.as_type(result, {:s, 32})
  end

  @doc """
  Integer division.
  Equivalent to math/div from glTF integer arithmetic specification.
  Truncates toward zero. Division by zero returns 0.
  Overflow case: -2147483648 / -1 = -2147483648.
  """
  def div(a, b, opts \\ []) do
    name = opts[:name] || "int_div"
    Axon.layer(&div_impl/3, [a, b], name: name)
  end

  defnp div_impl(a, b, _opts) do
    # Handle division by zero
    Nx.select(Nx.equal(b, 0), 0,
      Nx.divide(a, b) |> Nx.as_type({:s, 32}))
  end

  @doc """
  Integer remainder.
  Equivalent to math/rem from glTF integer arithmetic specification.
  Division by zero returns 0.
  """
  def rem(a, b, opts \\ []) do
    name = opts[:name] || "int_rem"
    Axon.layer(&rem_impl/3, [a, b], name: name)
  end

  defnp rem_impl(a, b, _opts) do
    # Handle division by zero case
    Nx.select(Nx.equal(b, 0), 0,
      Nx.remainder(a, b) |> Nx.as_type({:s, 32}))
  end

  @doc """
  Integer minimum.
  Equivalent to math/min from glTF integer arithmetic specification.
  """
  def min(a, b, opts \\ []) do
    name = opts[:name] || "int_min"
    Axon.layer(&min_impl/3, [a, b], name: name)
  end

  defnp min_impl(a, b, _opts) do
    Nx.min(a, b) |> Nx.as_type({:s, 32})
  end

  @doc """
  Integer maximum.
  Equivalent to math/max from glTF integer arithmetic specification.
  """
  def max(a, b, opts \\ []) do
    name = opts[:name] || "int_max"
    Axon.layer(&max_impl/3, [a, b], name: name)
  end

  defnp max_impl(a, b, _opts) do
    Nx.max(a, b) |> Nx.as_type({:s, 32})
  end

  @doc """
  Integer clamp.
  Equivalent to math/clamp from glTF integer arithmetic specification.
  """
  def clamp(a, min_val, max_val, opts \\ []) do
    name = opts[:name] || "int_clamp"
    Axon.layer(&clamp_impl/4, [a, min_val, max_val], name: name)
  end

  defnp clamp_impl(a, min_val, max_val, _opts) do
    Nx.max(Nx.min(a, max_val), min_val) |> Nx.as_type({:s, 32})
  end
end
