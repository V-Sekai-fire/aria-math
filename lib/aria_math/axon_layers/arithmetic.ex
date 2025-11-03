# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Arithmetic do
  @moduledoc """
  Basic arithmetic operations as Axon layers.
  Compatible with glTF interactivity specification.
  """

  import Nx.Defn

  @doc """
  Adds two values (component-wise for vectors/matrices).

  Equivalent to math/add from glTF interactivity specification.
  """
  def add(a, b, opts \\ []) do
    name = opts[:name] || "add"
    Axon.layer(&add_impl/3, [a, b], name: name)
  end

  defnp add_impl(a, b, _opts) do
    Nx.add(a, b)
  end

  @doc """
  Subtracts two values (component-wise for vectors/matrices).

  Equivalent to math/sub from glTF interactivity specification.
  """
  def sub(a, b, opts \\ []) do
    name = opts[:name] || "sub"
    Axon.layer(&sub_impl/3, [a, b], name: name)
  end

  defnp sub_impl(a, b, _opts) do
    Nx.subtract(a, b)
  end

  @doc """
  Multiplies two values (component-wise for vectors/matrices).

  Equivalent to math/mul from glTF interactivity specification.
  Note: This is element-wise multiplication, not matrix multiplication.
  For matrix multiplication, use multiply_matrices.
  """
  def mul(a, b, opts \\ []) do
    name = opts[:name] || "mul"
    Axon.layer(&mul_impl/3, [a, b], name: name)
  end

  defnp mul_impl(a, b, _opts) do
    Nx.multiply(a, b)
  end

  @doc """
  Divides two values (component-wise for vectors/matrices).

  Equivalent to math/div from glTF interactivity specification.
  """
  def div(a, b, opts \\ []) do
    name = opts[:name] || "div"
    Axon.layer(&div_impl/3, [a, b], name: name)
  end

  defnp div_impl(a, b, _opts) do
    Nx.divide(a, b)
  end
end
