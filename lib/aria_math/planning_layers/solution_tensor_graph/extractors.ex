# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.SolutionTensorGraph.Extractors do
  @moduledoc """
  Node and edge extraction functions for SolutionTensorGraph.

  This module contains functions responsible for extracting
  nodes and edges from solution tree structures.
  """

  @spec extract_nodes_and_edges(map()) :: {[map()], [map()]}
  def extract_nodes_and_edges(solution_tree) do
    {nodes, edges} = extract_nodes_and_edges_recursive(solution_tree, 0, -1)
    {nodes, edges}
  end

  @spec extract_nodes_and_edges_recursive(map(), non_neg_integer(), integer()) :: {[map()], [map()]}
  def extract_nodes_and_edges_recursive(node, node_id, parent_id) do
    # Create node entry
    node_entry = %{
      id: node_id,
      action: node.action,
      args: node.args || [],
      type:
        cond do
          # Root node is the goal
          parent_id == -1 -> :goal
          (node.children || []) == [] -> :primitive
          true -> :compound
        end
    }

    # Start with current node
    nodes = [node_entry]
    edges = []

    # Add edge from parent if not root
    edges =
      if parent_id >= 0 do
        [%{from: parent_id, to: node_id, type: :parent_child} | edges]
      else
        edges
      end

    # Process children recursively
    {child_nodes, child_edges, _} =
      Enum.reduce(node.children || [], {[], [], node_id + 1}, fn child, {acc_nodes, acc_edges, next_id} ->
        {child_nodes_result, child_edges_result} = extract_nodes_and_edges_recursive(child, next_id, node_id)
        {acc_nodes ++ child_nodes_result, acc_edges ++ child_edges_result, next_id + length(child_nodes_result)}
      end)

    {nodes ++ child_nodes, edges ++ child_edges}
  end
end
