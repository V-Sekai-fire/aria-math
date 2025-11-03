# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.Constants do
  @moduledoc """
  Mathematical constants as Axon layers for glTF interactivity compatibility.
  """

  import Nx.Defn

  @doc """
  Returns Euler's number (e ≈ 2.718281828459045).

  Equivalent to math/E from glTF interactivity specification.
  """
  def e(opts \\ []) do
    name = opts[:name] || "e"
    Axon.layer(&e_impl/1, [], name: name)
  end

  defnp e_impl(_opts) do
    # Euler's number (e ≈ 2.718281828459045)
    2.718281828459045
  end

  @doc """
  Returns the mathematical constant π (π ≈ 3.141592653589793).

  Equivalent to math/Pi from glTF interactivity specification.
  """
  def pi(opts \\ []) do
    name = opts[:name] || "pi"
    Axon.layer(&pi_impl/1, [], name: name)
  end

  defnp pi_impl(_opts) do
    Nx.PI
  end

  @doc """
  Returns positive infinity.

  Equivalent to math/Inf from glTF interactivity specification.
  """
  def inf(opts \\ []) do
    name = opts[:name] || "inf"
    Axon.layer(&inf_impl/1, [], name: name)
  end

  defnp inf_impl(_opts) do
    Nx.tensor(:infinity)
  end

  @doc """
  Returns Not a Number (NaN).

  Equivalent to math/NaN from glTF interactivity specification.
  """
  def nan(opts \\ []) do
    name = opts[:name] || "nan"
    Axon.layer(&nan_impl/1, [], name: name)
  end

  defnp nan_impl(_opts) do
    Nx.tensor(:nan)
  end
end
