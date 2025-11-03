# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.SolutionTensorGraph.Builders do
  @moduledoc """
  Tensor building functions for SolutionTensorGraph.

  This module contains all the functions responsible for building
  various tensor representations from solution tree data.
  """

  @spec build_adjacency_coo([map()], non_neg_integer()) :: {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}
  def build_adjacency_coo(edges, num_edges) do
    if num_edges == 0 do
      # Empty COO tensors
      {Nx.tensor([], type: :s64), Nx.tensor([], type: :s64), Nx.tensor([], type: :f32)}
    else
      # Extract row (from) and column (to) indices from edges
      row_indices = Nx.tensor(Enum.map(edges, & &1.from), type: :s64)
      col_indices = Nx.tensor(Enum.map(edges, & &1.to), type: :s64)
      # All edge values are 1.0 for directed graph adjacency
      edge_values = Nx.broadcast(1.0, {num_edges})

      {row_indices, col_indices, edge_values}
    end
  end

  @spec build_node_features([map()]) :: Nx.Tensor.t()
  def build_node_features(nodes) do
    num_nodes = length(nodes)

    if num_nodes == 0 do
      Nx.tensor([])
    else
      features =
        Enum.map(nodes, fn node ->
          # Create feature vector based on node properties
          # Normalize to 0-1
          action_hash = :erlang.phash2(node.action, 1000) / 1000.0
          # Normalize arg count
          args_count = length(node.args) / 10.0
          is_primitive = if node.type == :primitive, do: 1.0, else: 0.0

          # Create a basic feature vector (can be expanded for more sophisticated features)
          # Pad to 64 dimensions
          [action_hash, args_count, is_primitive] ++ List.duplicate(0.0, 61)
        end)

      Nx.tensor(features)
    end
  end

  @spec build_edge_features([map()]) :: Nx.Tensor.t()
  def build_edge_features(edges) do
    num_edges = length(edges)

    if num_edges == 0 do
      Nx.tensor([])
    else
      # Pad to 32 dimensions
      features = Enum.map(edges, fn _edge -> [1.0, 0.0, 0.0] ++ List.duplicate(0.0, 29) end)
      Nx.tensor(features)
    end
  end

  @spec build_node_types([map()]) :: Nx.Tensor.t()
  def build_node_types(nodes) do
    types =
      Enum.map(nodes, fn node ->
        case node[:type] do
          :task -> 0.0
          :method -> 1.0
          :action -> 2.0
          :goal -> 3.0
          _ -> -1.0
        end
      end)

    Nx.tensor(types)
  end

  @spec build_primitive_mask([map()]) :: Nx.Tensor.t()
  def build_primitive_mask(nodes) do
    mask =
      Enum.map(nodes, fn node ->
        if node.type == :primitive do
          1.0
        else
          0.0
        end
      end)

    Nx.tensor(mask)
  end

  @spec build_goal_mask([map()]) :: Nx.Tensor.t()
  def build_goal_mask(nodes) do
    mask =
      Enum.map(nodes, fn node ->
        if node[:type] == :goal do
          1.0
        else
          0.0
        end
      end)

    Nx.tensor(mask)
  end
end
