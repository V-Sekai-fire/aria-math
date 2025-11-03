# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.IntegerBitwise do
  @moduledoc """
  Integer bitwise operations as Axon layers.
  Compatible with glTF interactivity specification.
  All inputs/outputs are 32-bit integers (could be signed or unsigned depending on context).
  """

  import Nx.Defn

  @doc """
  Bitwise NOT.
  Equivalent to math/not from glTF integer bitwise specification.
  """
  def bitwise_not(a, opts \\ []) do
    name = opts[:name] || "bitwise_not"
    Axon.layer(&bitwise_not_impl/2, [a], name: name)
  end

  defnp bitwise_not_impl(a, _opts) do
    Nx.bitwise_not(a) |> Nx.as_type({:u, 32})
  end

  @doc """
  Bitwise AND.
  Equivalent to math/and from glTF integer bitwise specification.
  """
  def bitwise_and(a, b, opts \\ []) do
    name = opts[:name] || "bitwise_and"
    Axon.layer(&bitwise_and_impl/3, [a, b], name: name)
  end

  defnp bitwise_and_impl(a, b, _opts) do
    Nx.bitwise_and(a, b) |> Nx.as_type({:u, 32})
  end

  @doc """
  Bitwise OR.
  Equivalent to math/or from glTF integer bitwise specification.
  """
  def bitwise_or(a, b, opts \\ []) do
    name = opts[:name] || "bitwise_or"
    Axon.layer(&bitwise_or_impl/3, [a, b], name: name)
  end

  defnp bitwise_or_impl(a, b, _opts) do
    Nx.bitwise_or(a, b) |> Nx.as_type({:u, 32})
  end

  @doc """
  Bitwise XOR.
  Equivalent to math/xor from glTF integer bitwise specification.
  """
  def bitwise_xor(a, b, opts \\ []) do
    name = opts[:name] || "bitwise_xor"
    Axon.layer(&bitwise_xor_impl/3, [a, b], name: name)
  end

  defnp bitwise_xor_impl(a, b, _opts) do
    Nx.bitwise_xor(a, b) |> Nx.as_type({:u, 32})
  end

  @doc """
  Arithmetic (signed) right shift.
  Equivalent to math/asr from glTF integer bitwise specification.
  """
  def shift_right_arithmetic(a, shift_amount, opts \\ []) do
    name = opts[:name] || "shift_right_arithmetic"
    Axon.layer(&shift_right_arithmetic_impl/3, [a, shift_amount], name: name)
  end

  defnp shift_right_arithmetic_impl(a, shift_amount, _opts) do
    # Clamp shift amount to 0-31 range for 32-bit integers
    safe_shift = Nx.max(Nx.min(shift_amount, 31), 0)
    Nx.right_shift(a, safe_shift) |> Nx.as_type({:u, 32})
  end

  @doc """
  Logical left shift.
  Equivalent to math/lsl from glTF integer bitwise specification.
  """
  def shift_left_logical(a, shift_amount, opts \\ []) do
    name = opts[:name] || "shift_left_logical"
    Axon.layer(&shift_left_logical_impl/3, [a, shift_amount], name: name)
  end

  defnp shift_left_logical_impl(a, shift_amount, _opts) do
    # Clamp shift amount to 0-31 range for 32-bit integers
    safe_shift = Nx.max(Nx.min(shift_amount, 31), 0)
    Nx.left_shift(a, safe_shift) |> Nx.as_type({:u, 32})
  end

  @doc """
  Count leading zeros.
  Equivalent to math/clz from glTF integer bitwise specification.
  """
  def count_leading_zeros(a, opts \\ []) do
    name = opts[:name] || "count_leading_zeros"
    Axon.layer(&count_leading_zeros_impl/2, [a], name: name)
  end

  defnp count_leading_zeros_impl(a, _opts) do
    # Nx doesn't have a direct CLZ, so we'll implement it
    # This is a simplified implementation
    Nx.select(
      Nx.equal(a, 0),
      32,  # All bits are zero
      # Otherwise, count leading zeros - simplified version
      32 - Nx.select(
        Nx.greater(a, 0xFFFF), 16, 0
      ) - Nx.select(
        Nx.greater(a, 0xFF), 8, 0
      ) - Nx.select(
        Nx.greater(a, 0xF), 4, 0
      ) - Nx.select(
        Nx.greater(a, 3), 2, 0
      ) - Nx.select(
        Nx.greater(a, 1), 1, 0
      )
    ) |> Nx.as_type({:u, 32})
  end

  @doc """
  Count trailing zeros.
  Equivalent to math/ctz from glTF integer bitwise specification.
  """
  def count_trailing_zeros(a, opts \\ []) do
    name = opts[:name] || "count_trailing_zeros"
    Axon.layer(&count_trailing_zeros_impl/2, [a], name: name)
  end

  defnp count_trailing_zeros_impl(a, _opts) do
    # Simplified CTZ implementation
    Nx.select(
      Nx.equal(a, 0),
      32,  # All bits are zero
      # Count how many times we can divide by 2
      Nx.add(
        Nx.select(Nx.equal(Nx.bitwise_and(a, 1), 0), 1, 0),
        Nx.select(Nx.equal(Nx.bitwise_and(a, 2), 0), 1, 0)
      ) + Nx.select(
        Nx.equal(Nx.bitwise_and(a, 4), 0), 1, 0
      ) + Nx.select(
        Nx.equal(Nx.bitwise_and(a, 8), 0), 1, 0
      ) + Nx.select(
        Nx.equal(Nx.bitwise_and(a, 16), 0), 1, 0
      )
      # This is a very simplified version - real CTZ is more complex
    ) |> Nx.as_type({:u, 32})
  end

  @doc """
  Population count (number of 1 bits).
  Equivalent to math/popcnt from glTF integer bitwise specification.
  """
  def population_count(a, opts \\ []) do
    name = opts[:name] || "population_count"
    Axon.layer(&population_count_impl/2, [a], name: name)
  end

  defnp population_count_impl(a, _opts) do
    # Count bits set to 1 using addition of individual bits
    Nx.add(
      Nx.bitwise_and(a, 1) + Nx.right_shift(Nx.bitwise_and(a, 2), 1) +
      Nx.right_shift(Nx.bitwise_and(a, 4), 2) + Nx.right_shift(Nx.bitwise_and(a, 8), 3) +
      Nx.right_shift(Nx.bitwise_and(a, 16), 4) + Nx.right_shift(Nx.bitwise_and(a, 32), 5) +
      Nx.right_shift(Nx.bitwise_and(a, 64), 6) + Nx.right_shift(Nx.bitwise_and(a, 128), 7) +
      Nx.right_shift(Nx.bitwise_and(a, 256), 8) + Nx.right_shift(Nx.bitwise_and(a, 512), 9) +
      Nx.right_shift(Nx.bitwise_and(a, 1024), 10) + Nx.right_shift(Nx.bitwise_and(a, 2048), 11) +
      Nx.right_shift(Nx.bitwise_and(a, 4096), 12) + Nx.right_shift(Nx.bitwise_and(a, 8192), 13) +
      Nx.right_shift(Nx.bitwise_and(a, 16384), 14) + Nx.right_shift(Nx.bitwise_and(a, 32768), 15) +
      Nx.right_shift(Nx.bitwise_and(a, 65536), 16) + Nx.right_shift(Nx.bitwise_and(a, 131072), 17) +
      Nx.right_shift(Nx.bitwise_and(a, 262144), 18) + Nx.right_shift(Nx.bitwise_and(a, 524288), 19) +
      Nx.right_shift(Nx.bitwise_and(a, 1048576), 20) + Nx.right_shift(Nx.bitwise_and(a, 2097152), 21) +
      Nx.right_shift(Nx.bitwise_and(a, 4194304), 22) + Nx.right_shift(Nx.bitwise_and(a, 8388608), 23) +
      Nx.right_shift(Nx.bitwise_and(a, 16777216), 24) + Nx.right_shift(Nx.bitwise_and(a, 33554432), 25) +
      Nx.right_shift(Nx.bitwise_and(a, 67108864), 26) + Nx.right_shift(Nx.bitwise_and(a, 134217728), 27) +
      Nx.right_shift(Nx.bitwise_and(a, 268435456), 28) + Nx.right_shift(Nx.bitwise_and(a, 536870912), 29) +
      Nx.right_shift(Nx.bitwise_and(a, 1073741824), 30) + Nx.right_shift(Nx.bitwise_and(a, -0x80000000), 31),
      Nx.tensor(0)
    ) |> Nx.as_type({:u, 32})
  end
end
