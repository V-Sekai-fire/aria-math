# AriaMath

A comprehensive mathematical operations library for 3D graphics and computational geometry in Elixir. AriaMath provides high-performance implementations of vectors, quaternions, matrices, and geometric primitives with GPU acceleration support via Nx and TorchX.

## Features

- **Vector3**: 3D vector operations with arithmetic, utilities, and tensor support
- **Quaternion**: Quaternion operations, conversions, and utilities for 3D rotations
- **Matrix4**: 4x4 matrix operations for transformations, Euler angles, and batch processing
- **Primitives**: Geometric primitives including spheres, cylinders, and mathematical utilities
- **GPU Acceleration**: Leverages Nx and TorchX for high-performance tensor operations
- **Memory Management**: Efficient memory handling for large-scale computations

## Installation

Add `aria_math` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:aria_math, "~> 0.1.0"}
  ]
end
```

## Usage

```elixir
# Vector operations
vector1 = AriaMath.Vector3.new(1.0, 2.0, 3.0)
vector2 = AriaMath.Vector3.new(4.0, 5.0, 6.0)
result = AriaMath.Vector3.add(vector1, vector2)

# Quaternion operations
quat = AriaMath.Quaternion.from_euler(0.0, 1.57, 0.0)
rotated = AriaMath.Quaternion.rotate_vector(quat, vector1)

# Matrix transformations
matrix = AriaMath.Matrix4.translation(1.0, 2.0, 3.0)
transformed = AriaMath.Matrix4.transform_point(matrix, vector1)
```

## Documentation

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/aria_math>.

## License

Copyright (c) 2025-present K. S. Ernest (iFire) Lee

This project is licensed under the MIT License.
