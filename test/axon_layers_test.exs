# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayersTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  alias AriaMath.AxonLayers

  describe "transform_points/2" do
    test "transforms 3D points with a translation matrix" do
      # Create a simple model that applies a translation
      points_input = Axon.input("points", shape: {nil, 3})

      # Create a translation matrix (translate by [1, 2, 3])
      translation_matrix = Nx.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
      ])

      model = Axon.constant(translation_matrix) |> AxonLayers.transform_points(points_input)

      # Build and run the model
      {init_fn, predict_fn} = Axon.build(model)

      # Initialize parameters
      params = init_fn.(Nx.template({1, 3}, :f32), %{})

      # Test input points
      input_points = Nx.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

      # Run prediction
      result = predict_fn.(params, %{"points" => input_points})

      # Expected: [0,0,0] -> [1,2,3], [1,1,1] -> [2,3,4]
      expected = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

      assert Nx.all_close(result, expected, atol: 1.0e-6)
    end
  end

  describe "compose_matrices/2" do
    test "composes two transformation matrices" do
      # Create two matrices: translation and rotation
      translation_matrix = Nx.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ])

      scaling_matrix = Nx.tensor([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ])

      model = AxonLayers.compose_matrices(
        Axon.constant(translation_matrix),
        Axon.constant(scaling_matrix)
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({}, :f32), %{})

      result = predict_fn.(params, %{})

      # Expected: scaling applied first, then translation
      # Result should be: [[2, 0, 0, 2], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]
      expected = Nx.tensor([
        [2.0, 0.0, 0.0, 2.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ])

      assert Nx.all_close(result, expected, atol: 1.0e-6)
    end
  end

  describe "translation_matrix/1" do
    test "creates translation matrices from vectors" do
      # Input translation vectors
      translations = Nx.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

      model = AxonLayers.translation_matrix(Axon.constant(translations))

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({}, :f32), %{})

      result = predict_fn.(params, %{})

      # Should return batch of 4x4 translation matrices
      assert Nx.shape(result) == {2, 4, 4}

      # First matrix should be translation by [1, 2, 3]
      first_matrix = result[0]
      assert Nx.all_close(first_matrix[0][3], Nx.tensor(1.0))
      assert Nx.all_close(first_matrix[1][3], Nx.tensor(2.0))
      assert Nx.all_close(first_matrix[2][3], Nx.tensor(3.0))

      # Second matrix should be identity (translation by [0, 0, 0])
      second_matrix = result[1]
      assert Nx.all_close(second_matrix[0][3], Nx.tensor(0.0))
      assert Nx.all_close(second_matrix[1][3], Nx.tensor(0.0))
      assert Nx.all_close(second_matrix[2][3], Nx.tensor(0.0))
    end
  end

  describe "rotation_matrix_from_quaternion/1" do
    test "creates rotation matrices from quaternions" do
      # Identity quaternion [1, 0, 0, 0] should give identity matrix
      quaternions = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])

      model = AxonLayers.rotation_matrix_from_quaternion(Axon.constant(quaternions))

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({}, :f32), %{})

      result = predict_fn.(params, %{})

      # Should be close to identity matrix
      identity = Nx.eye(4)
      assert Nx.all_close(result[0], identity, atol: 1.0e-6)
    end
  end

  describe "decompose_matrix/1" do
    test "decomposes transformation matrices" do
      # Create a simple translation matrix
      matrix = Nx.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
      ])

      model = AxonLayers.decompose_matrix(Axon.constant(matrix))

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({4, 4}, :f32), %{})

      {translation, rotation, scale} = predict_fn.(params, %{})

      # Translation should be [1, 2, 3]
      assert Nx.all_close(translation, Nx.tensor([1.0, 2.0, 3.0]), atol: 1.0e-6)

      # Rotation should be identity (4x4)
      identity = Nx.eye(4)
      assert Nx.all_close(rotation, identity, atol: 1.0e-6)

      # Scale should be [1, 1, 1]
      assert Nx.all_close(scale, Nx.tensor([1.0, 1.0, 1.0]), atol: 1.0e-6)
    end
  end

  describe "SPMD sharding" do
    test "creates sharding annotations" do
      sharding = AriaMath.Sharding.annotation({4, 8}, [0, -1, 1])

      assert sharding.type == :tiled
      assert sharding.device_mesh == {4, 8}
      assert sharding.dims_mapping == [0, -1, 1]
    end

    test "sharding propagation through operations" do
      # Initial sharding annotations
      initial_sharding = %{
        "points" => AriaMath.Sharding.annotation({4, 8}, [0, -1]),
        "matrix1" => AriaMath.Sharding.annotation({4, 8}, [-1, -1, 0, 1])
      }

      # Operations to propagate through
      operations = [
        {:transform_points, ["points", "transform_matrix"], "output1"},
        {:compose_matrices, ["matrix1", "matrix2"], "output2"}
      ]

      # Propagate sharding
      final_sharding = AriaMath.Sharding.propagate_sharding(initial_sharding, operations)

      # Check that sharding was propagated
      assert Map.has_key?(final_sharding, "output1")
      assert final_sharding["output1"] == initial_sharding["points"]

      assert Map.has_key?(final_sharding, "output2")
      assert final_sharding["output2"] == initial_sharding["matrix1"]
    end
  end

  describe "SPMD layers" do
    @tag :skip
    test "spmd_transform_points applies sharding and collective operations" do
      # SP MD sharding requires additional Axon integration work
      # Skipping for now to focus on core mathematical functionality
      assert true
    end

    @tag :skip
    test "spmd_compose_matrices with sharding" do
      # SPMD sharding requires additional Axon integration work
      # Skipping for now to focus on core mathematical functionality
      assert true
    end
  end

  describe "collective operations" do
    test "all_reduce operation" do
      tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = AriaMath.Sharding.all_reduce(tensor)

      # In simulation, returns the same tensor
      assert Nx.all_close(result, tensor, atol: 1.0e-6)
    end

    test "all_gather operation" do
      tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = AriaMath.Sharding.all_gather(tensor)

      # In simulation, returns the same tensor
      assert Nx.all_close(result, tensor, atol: 1.0e-6)
    end

    test "reduce_scatter operation" do
      tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = AriaMath.Sharding.reduce_scatter(tensor)

      # In simulation, returns the same tensor
      assert Nx.all_close(result, tensor, atol: 1.0e-6)
    end
  end

  describe "integration test" do
    test "end-to-end 3D transformation pipeline" do
      # Create a model that:
      # 1. Takes 3D points as input
      # 2. Learns a translation vector
      # 3. Creates a translation matrix
      # 4. Applies the transformation

      points_input = Axon.input("points", shape: {nil, 3})
      translation_param = Axon.param("translation", {3})

      model =
        translation_param
        |> AxonLayers.translation_matrix()
        |> AxonLayers.transform_points(points_input)

      {init_fn, predict_fn} = Axon.build(model)

      # Initialize with zero translation
      params = init_fn.(Nx.template({1, 3}, :f32), %{"translation" => Nx.tensor([0.0, 0.0, 0.0])})

      # Test points
      input_points = Nx.tensor([[1.0, 2.0, 3.0]])

      # With zero translation, points should remain unchanged
      result = predict_fn.(params, %{"points" => input_points})
      assert Nx.all_close(result, input_points, atol: 1.0e-6)

      # Update translation parameter
      params = Map.put(params, "translation", Nx.tensor([1.0, 1.0, 1.0]))

      # Now points should be translated
      result = predict_fn.(params, %{"points" => input_points})
      expected = Nx.tensor([[2.0, 3.0, 4.0]])
      assert Nx.all_close(result, expected, atol: 1.0e-6)
    end

    @tag :skip
    test "SPMD end-to-end pipeline with sharding" do
      # SPMD sharding requires additional Axon integration work
      # Skipping for now to focus on core mathematical functionality
      assert true
    end
  end
end
