# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.SolutionTensorGraph do
  @moduledoc """
  Tensor-based graph representation for planning solutions.

  This module converts traditional solution trees into tensor format
  for efficient GPU/CPU processing using Nx tensors.

  ## Overview

  SolutionTensorGraph provides a unified tensor representation for planning solutions,
  enabling efficient analysis, optimization, and execution of plans. It converts
  hierarchical solution trees into graph structures suitable for tensor operations.

  ## Architecture

  The graph uses multiple tensor types:
  - **Adjacency Tensors**: COO format for sparse graph representation
  - **Node Features**: Multi-dimensional feature vectors for each node
  - **Edge Features**: Feature vectors describing relationships between nodes
  - **Type Masks**: Boolean tensors identifying node/edge types

  ## Node Types

  - `:goal` - Root goal node
  - `:compound` - Compound task nodes (decomposable)
  - `:primitive` - Primitive action nodes (executable)

  ## Usage

  ### Creating from Solution Tree
  ```elixir
  solution_tree = %{root: %{type: :compound, subtasks: [...]}}
  graph = SolutionTensorGraph.from_solution_tree(solution_tree)
  ```

  ### Analyzing Execution State
  ```elixir
  complete? = SolutionTensorGraph.execution_complete?(graph)
  primitive_nodes = SolutionTensorGraph.get_primitive_nodes(graph)
  ```

  ### Graph Operations
  ```elixir
  neighbors = SolutionTensorGraph.get_neighbors(graph, [node_id])
  features = SolutionTensorGraph.get_node_features(graph, [node_id])
  ```
  """
  @enforce_keys [:num_nodes, :num_edges, :row_indices, :col_indices, :edge_values, :node_features, :edge_features]
  defstruct [
    :num_nodes,
    :num_edges,
    :row_indices,
    :col_indices,
    :edge_values,
    :node_features,
    :edge_features,
    :node_types,
    :primitive_mask,
    :goal_mask,
    :metadata
  ]

  @type t :: %__MODULE__{
          num_nodes: non_neg_integer(),
          num_edges: non_neg_integer(),
          row_indices: Nx.Tensor.t(),
          col_indices: Nx.Tensor.t(),
          edge_values: Nx.Tensor.t(),
          node_features: Nx.Tensor.t(),
          edge_features: Nx.Tensor.t(),
          node_types: Nx.Tensor.t(),
          primitive_mask: Nx.Tensor.t(),
          goal_mask: Nx.Tensor.t(),
          metadata: map()
        }

  @doc "Create a SolutionTensorGraph from a solution tree.

Converts the traditional nested map structure into tensor format.
"
  @spec from_solution_tree(map()) :: t()
  def from_solution_tree(solution_tree) do
    alias AriaMath.AxonLayers.SolutionTensorGraph.{Extractors, Builders}

    # Extract nodes and edges from solution tree
    {nodes, edges} = Extractors.extract_nodes_and_edges(solution_tree)

    num_nodes = length(nodes)
    num_edges = length(edges)

    # Build tensor representations
    {row_indices, col_indices, edge_values} = Builders.build_adjacency_coo(edges, num_edges)
    node_features = Builders.build_node_features(nodes)
    edge_features = Builders.build_edge_features(edges)
    node_types = Builders.build_node_types(nodes)
    primitive_mask = Builders.build_primitive_mask(nodes)
    goal_mask = Builders.build_goal_mask(nodes)

    metadata = %{
      num_nodes: num_nodes,
      num_edges: num_edges,
      nodes: nodes,
      edges: edges
    }

    %__MODULE__{
      num_nodes: num_nodes,
      num_edges: num_edges,
      row_indices: row_indices,
      col_indices: col_indices,
      edge_values: edge_values,
      node_features: node_features,
      edge_features: edge_features,
      node_types: node_types,
      primitive_mask: primitive_mask,
      goal_mask: goal_mask,
      metadata: metadata
    }
  end

  @doc "Convert SolutionTensorGraph back to traditional solution tree format.

Useful for compatibility with existing code that expects map-based structures.
"
  @spec to_solution_tree(t()) :: map()
  def to_solution_tree(_solution_tensor_graph) do
    raise "TODO: Implement #{__MODULE__}.to_solution_tree"
  end

  @doc "Get node features for specific node indices.

Efficiently extracts features for a subset of nodes.
"
  @spec get_node_features(t(), [non_neg_integer()]) :: Nx.Tensor.t()
  def get_node_features(%__MODULE__{node_features: node_features}, node_indices) do
    if Enum.empty?(node_indices) do
      Nx.tensor([])
    else
      # Index into the feature tensor
      indices = Nx.tensor(node_indices)
      Nx.take(node_features, indices)
    end
  end

  @doc "Get neighbors of specified nodes.

Returns indices of nodes connected to the given nodes.
"
  @spec get_neighbors(t(), [non_neg_integer()]) :: [non_neg_integer()]
  def get_neighbors(%__MODULE__{row_indices: row_indices, col_indices: col_indices}, node_indices) do
    # Find all edges where from_node is in node_indices
    row_list = Nx.to_flat_list(row_indices)
    col_list = Nx.to_flat_list(col_indices)

    for i <- node_indices, {from_idx, to_idx} <- Enum.zip(row_list, col_list), from_idx == i do
      to_idx
    end
    |> Enum.uniq()
  end

  @doc "Check if graph execution is complete.

Determines if all primitive actions have been executed.
"
  @spec execution_complete?(t()) :: boolean()
  def execution_complete?(_solution_tensor_graph) do
    raise "TODO: Implement #{__MODULE__}.execution_complete?"
  end

  @doc "Get primitive action nodes.

Returns indices of nodes that represent executable actions.
"
  @spec get_primitive_nodes(t()) :: [non_neg_integer()]
  def get_primitive_nodes(%__MODULE__{primitive_mask: primitive_mask}) do
    # Find indices where primitive_mask is 1.0
    primitive_mask
    |> Nx.to_flat_list()
    |> Enum.with_index()
    |> Enum.filter(fn {value, _index} -> value == 1.0 end)
    |> Enum.map(fn {_value, index} -> index end)
  end

  @doc "Get goal nodes.

Returns indices of nodes that represent goal states.
"
  @spec get_goal_nodes(t()) :: [non_neg_integer()]
  def get_goal_nodes(_solution_tensor_graph) do
    raise "TODO: Implement #{__MODULE__}.get_goal_nodes"
  end

  @doc "Update node features.

Modifies features for specified nodes.
"
  @spec update_node_features(t(), [non_neg_integer()], Nx.Tensor.t()) :: t()
  def update_node_features(_solution_tensor_graph, _node_indices, _new_features) do
    raise "TODO: Implement #{__MODULE__}.update_node_features"
  end
end
