# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule SolutionTensorGraphTest do
  use ExUnit.Case, async: true

  alias AriaMath.AxonLayers.SolutionTensorGraph

  describe "Survival Craft torch graph operations" do
    test "creates torch graph from Survival Craft planning solution" do
      # Create a solution tree manually for testing
      solution_tree = %{
        action: :establish_base_camp,
        args: ["test_camp"],
        children: [
          %{
            action: :gather_wood_task,
            args: [10],
            children: []
          },
          %{
            action: :build_shelter_task,
            args: ["test_camp"],
            children: []
          }
        ]
      }

      # Create torch graph from solution tree
      torch_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

      assert %SolutionTensorGraph{} = torch_graph
      # Root + 2 children
      assert torch_graph.num_nodes == 3
      # 2 parent-child relationships
      assert torch_graph.num_edges == 2

      # Should have adjacency matrix
      assert is_struct(torch_graph.row_indices, Nx.Tensor)
      assert is_struct(torch_graph.col_indices, Nx.Tensor)
      assert is_struct(torch_graph.edge_values, Nx.Tensor)

      # Should have node and edge features
      assert is_struct(torch_graph.node_features, Nx.Tensor)
      assert is_struct(torch_graph.edge_features, Nx.Tensor)

      # Should have metadata
      assert is_map(torch_graph.metadata)
      assert Map.has_key?(torch_graph.metadata, :created_at)
    end

    test "torch graph represents Survival Craft activity dependencies" do
      # Create a more complex solution tree
      solution_tree = %{
        action: :survive_first_night,
        args: [],
        children: [
          %{
            action: :find_water_task,
            args: ["nearby"],
            children: []
          },
          %{
            action: :build_shelter_task,
            args: ["emergency"],
            children: []
          },
          %{
            action: :rest_task,
            args: ["overnight"],
            children: []
          }
        ]
      }

      torch_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

      # Graph should represent the dependency structure
      # Root + 3 children
      assert torch_graph.num_nodes == 4
      # 3 parent-child relationships
      assert torch_graph.num_edges == 3

      # Should have proper node features
      # Default feature size
      assert torch_graph.node_features.shape == {4, 64}

      # Should have edge features
      # Default edge feature size
      assert torch_graph.edge_features.shape == {3, 32}
    end

    test "torch graph execution tracking for Survival Craft plans" do
      solution_tree = %{
        action: :gather_resources,
        args: [["wood", "water"]],
        children: [
          %{action: :gather_wood_task, args: [5], children: []},
          %{action: :find_water_task, args: ["source"], children: []}
        ]
      }

      torch_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

      # Test execution completion checking
      assert not SolutionTensorGraph.execution_complete?(torch_graph)

      # Get primitive nodes (leaf actions)
      primitive_nodes = SolutionTensorGraph.get_primitive_nodes(torch_graph)
      assert length(primitive_nodes) == 2

      # Get goal nodes (root goals)
      goal_nodes = SolutionTensorGraph.get_goal_nodes(torch_graph)
      assert length(goal_nodes) == 1
    end

    test "torch graph node feature extraction for Survival Craft actions" do
      solution_tree = %{
        action: :craft_tool,
        args: [],
        children: [
          %{action: :gather_materials_task, args: [["wood", "stone"]], children: []}
        ]
      }

      torch_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

      # Extract node features for specific nodes
      all_nodes = 0..(torch_graph.num_nodes - 1) |> Enum.to_list()
      node_features = SolutionTensorGraph.get_node_features(torch_graph, all_nodes)

      assert node_features.shape == {torch_graph.num_nodes, 64}

      # Get neighbors for root node
      root_neighbors = SolutionTensorGraph.get_neighbors(torch_graph, [0])
      # One child
      assert length(root_neighbors) == 1
    end

    test "torch graph conversion round-trip for Survival Craft plans" do
      original_tree = %{
        action: :signal_rescue,
        args: [],
        children: [
          %{action: :build_signal_fire_task, args: ["large"], children: []},
          %{action: :send_distress_signal_task, args: ["repeated"], children: []}
        ]
      }

      # Convert to torch graph
      torch_graph = SolutionTensorGraph.from_solution_tree(original_tree)

      # Convert back to solution tree
      reconstructed_tree = SolutionTensorGraph.to_solution_tree(torch_graph)

      # Should preserve the structure
      assert reconstructed_tree.action == :signal_rescue
      assert length(reconstructed_tree.children) == 2
    end

    test "torch graph handles Survival Craft complex planning scenarios" do
      # Complex multi-level planning
      solution_tree = %{
        action: :establish_settlement,
        args: ["permanent"],
        children: [
          %{
            action: :establish_base_camp,
            args: ["initial"],
            children: [
              %{action: :gather_wood_task, args: [20], children: []},
              %{action: :build_shelter_task, args: ["initial"], children: []}
            ]
          },
          %{
            action: :expand_settlement,
            args: ["permanent"],
            children: [
              %{action: :craft_advanced_tools_task, args: [], children: []},
              %{action: :build_workshop_task, args: ["permanent"], children: []}
            ]
          }
        ]
      }

      torch_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

      # Should handle complex dependency graphs
      # Root + 2 intermediate + 4 leaves
      assert torch_graph.num_nodes == 7
      # Parent-child relationships
      assert torch_graph.num_edges == 6

      # Should have COO format adjacency (sparse)
      assert is_struct(torch_graph.row_indices, Nx.Tensor)
      assert is_struct(torch_graph.col_indices, Nx.Tensor)
      assert is_struct(torch_graph.edge_values, Nx.Tensor)
      assert Nx.shape(torch_graph.row_indices) == {6}
      assert Nx.shape(torch_graph.col_indices) == {6}
      assert Nx.shape(torch_graph.edge_values) == {6}
    end

    test "torch graph metadata includes Survival Craft planning context" do
      # Plan with specific context
      solution_tree = %{
        action: :survive_first_night,
        args: [],
        children: [
          %{action: :find_water_task, args: ["emergency"], children: []},
          %{action: :build_shelter_task, args: ["quick"], children: []}
        ]
      }

      torch_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

      # Metadata should include planning context
      metadata = torch_graph.metadata

      assert Map.has_key?(metadata, :created_at)
      assert Map.has_key?(metadata, :version)
      assert Map.has_key?(metadata, :original_tree)

      # Original tree should be preserved
      assert metadata.original_tree.action == :survive_first_night
      assert length(metadata.original_tree.children) == 2
    end

    test "torch graph node feature updates for Survival Craft plan modifications" do
      # Create a proper solution tree with children
      solution_tree = %{
        action: :gather_resources,
        args: ["wood"],
        children: [
          %{action: :gather_wood_task, args: [5], children: []},
          %{action: :find_water_task, args: ["source"], children: []}
        ]
      }

      torch_graph = SolutionTensorGraph.from_solution_tree(solution_tree)

      # Should have 3 nodes: root + 2 children
      assert torch_graph.num_nodes == 3
      assert torch_graph.num_edges == 2

      # COO format adjacency should show connections
      assert is_struct(torch_graph.row_indices, Nx.Tensor)
      assert is_struct(torch_graph.col_indices, Nx.Tensor)
      assert is_struct(torch_graph.edge_values, Nx.Tensor)
      assert Nx.shape(torch_graph.row_indices) == {2}
      assert Nx.shape(torch_graph.col_indices) == {2}
      assert Nx.shape(torch_graph.edge_values) == {2}

      # Root (node 0) should connect to children (nodes 1 and 2)
      row_indices = Nx.to_list(torch_graph.row_indices)
      col_indices = Nx.to_list(torch_graph.col_indices)
      # Root node has outgoing edges
      assert 0 in row_indices
      # Node 1 is a target
      assert 1 in col_indices
      # Node 2 is a target
      assert 2 in col_indices

      # Get original features
      original_features = SolutionTensorGraph.get_node_features(torch_graph, [0])

      # Update node features
      # Updated feature vector
      new_features = Nx.tensor([[1.0, 0.5, 0.8, 0.2]])
      updated_graph = SolutionTensorGraph.update_node_features(torch_graph, [0], new_features)

      # Verify features were updated
      updated_features = SolutionTensorGraph.get_node_features(updated_graph, [0])

      # Features should be different after update
      assert Nx.to_number(Nx.all(Nx.equal(original_features, updated_features))) == 0

      # Verify COO format is preserved
      assert is_struct(updated_graph.row_indices, Nx.Tensor)
      assert is_struct(updated_graph.col_indices, Nx.Tensor)
      assert is_struct(updated_graph.edge_values, Nx.Tensor)
    end
  end
end
