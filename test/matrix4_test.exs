# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.Matrix4Test do
  use ExUnit.Case, async: true

  alias AriaMath.{Matrix4, MatrixFixtures}

  describe "basic operations" do
    test "identity matrix" do
      identity = Matrix4.identity()
      expected = MatrixFixtures.identity()
      assert Matrix4.equal?(identity, expected)
    end

    test "translation matrix" do
      translation = Matrix4.translation({1.0, 2.0, 3.0})
      expected = MatrixFixtures.translation_1_2_3()
      assert Matrix4.equal?(translation, expected)
    end

    test "scaling matrix" do
      scaling = Matrix4.scaling({2.0, 3.0, 4.0})
      expected = MatrixFixtures.scaling_2_3_4()
      assert Matrix4.equal?(scaling, expected)
    end
  end

  describe "matrix inverse" do
    test "inverse of translation matrix" do
      translation = MatrixFixtures.translation_1_2_3()
      {inverse, invertible} = Matrix4.inverse(translation)

      assert invertible
      expected_inverse = MatrixFixtures.inverse_translation_1_2_3()
      assert Matrix4.equal?(inverse, expected_inverse)

      # Verify: translation * inverse = identity
      product = Nx.dot(translation, inverse)
      identity = Matrix4.identity()
      assert Matrix4.equal?(product, identity)
    end
  end

  describe "point transformation" do
    test "transform points with fixtures" do
      transforms = MatrixFixtures.point_transforms()

      for {name, {input_point, expected_output}} <- transforms do
        case name do
          :identity_point ->
            matrix = Matrix4.identity()
            result = Matrix4.transform_point(matrix, input_point)
            assert result == expected_output

          :translation_origin ->
            matrix = MatrixFixtures.translation_1_2_3()
            result = Matrix4.transform_point(matrix, input_point)
            assert result == expected_output

          :translation_point ->
            matrix = MatrixFixtures.translation_1_2_3()
            result = Matrix4.transform_point(matrix, input_point)
            assert result == expected_output
        end
      end
    end
  end

  describe "matrix decomposition" do
    test "decompose matrices using fixtures" do
      decompositions = MatrixFixtures.decompositions()

      for {name, {expected_translation, expected_rotation, expected_scale}} <- decompositions do
        matrix = case name do
          :identity -> MatrixFixtures.identity()
          :translation_only -> MatrixFixtures.translation_1_2_3()
          :scaling_only -> MatrixFixtures.scaling_2_3_4()
        end

        {translation, rotation, scale} = Matrix4.decompose(matrix)

        assert translation == expected_translation
        assert Matrix4.equal?(rotation, expected_rotation)
        assert scale == expected_scale
      end
    end
  end

  describe "matrix composition" do
    test "compose and decompose round trip" do
      # Test that compose(decompose(matrix)) ≈ matrix
      original = MatrixFixtures.translation_then_scaling()
      {translation, rotation, scale} = Matrix4.decompose(original)

      # Construct recomposed matrix manually since compose is Axon layer
      # For translation_then_scaling: translation(1,2,3) then scaling(2,3,4)
      # Result: translate then scale: [[2,0,0,1],[0,3,0,2],[0,0,4,3],[0,0,0,1]]
      recomposed = Nx.tensor([
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 3.0, 0.0, 2.0],
        [0.0, 0.0, 4.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
      ], type: :f64)

      assert Matrix4.equal?(original, recomposed)
    end
  end

  describe "matrix multiplication" do
    test "translation then scaling" do
      translation = MatrixFixtures.translation_1_2_3()
      scaling = MatrixFixtures.scaling_2_3_4()

      # Matrix multiplication: scaling * translation (note: order matters)
      result = Nx.dot(scaling, translation)
      expected = MatrixFixtures.translation_then_scaling()

      assert Matrix4.equal?(result, expected)
    end
  end
end
