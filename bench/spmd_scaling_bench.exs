# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.SPMDScalingBench do
  @moduledoc """
  Benchee benchmarks for SPMD scaling performance measurement.

  This benchmark suite measures the scaling characteristics of AriaMath's SPMD
  implementation across different device counts and workloads.
  """

  def run do
    # Run the benchmarks
    Benchee.run(
      %{
        "SPMD Process Scaling (1 device)" => fn -> spmd_scaling_benchmark(1) end,
        "SPMD Process Scaling (2 devices)" => fn -> spmd_scaling_benchmark(2) end,
        "SPMD Process Scaling (4 devices)" => fn -> spmd_scaling_benchmark(4) end,
        "SPMD Process Scaling (8 devices)" => fn -> spmd_scaling_benchmark(8) end
      },
      time: 10,
      memory_time: 2,
      reduction_time: 2,
      formatters: [
        Benchee.Formatters.Console
      ],
      save: [path: "bench/spmd_scaling.benchee"],
      load: "bench/spmd_scaling.benchee"
    )
  end

  def run_memory_scaling do
    Benchee.run(
      %{
        "Memory Scaling (batch=32, 1 device)" => fn -> memory_scaling_benchmark(32, 1) end,
        "Memory Scaling (batch=32, 2 devices)" => fn -> memory_scaling_benchmark(32, 2) end,
        "Memory Scaling (batch=32, 4 devices)" => fn -> memory_scaling_benchmark(32, 4) end,
        "Memory Scaling (batch=64, 1 device)" => fn -> memory_scaling_benchmark(64, 1) end,
        "Memory Scaling (batch=64, 2 devices)" => fn -> memory_scaling_benchmark(64, 2) end,
        "Memory Scaling (batch=64, 4 devices)" => fn -> memory_scaling_benchmark(64, 4) end,
        "Memory Scaling (batch=128, 1 device)" => fn -> memory_scaling_benchmark(128, 1) end,
        "Memory Scaling (batch=128, 2 devices)" => fn -> memory_scaling_benchmark(128, 2) end,
        "Memory Scaling (batch=128, 4 devices)" => fn -> memory_scaling_benchmark(128, 4) end
      },
      time: 5,
      memory_time: 5,
      formatters: [
        Benchee.Formatters.Console
      ]
    )
  end

  def run_backend_scaling do
    # Test scaling across different detected backends (excluding CPU since we have process-based scaling)
    backends = AriaMath.BackendDetector.detect_backends() |> Enum.reject(&(&1 == :cpu))

    # Create benchmark cases for each available backend
    benchmark_cases = for backend <- backends, into: %{} do
      [
        {"#{backend} SPMD (batch=64, 2 shards)", fn -> backend_spmd_benchmark(backend, 64, 2) end},
        {"#{backend} SPMD (batch=128, 2 shards)", fn -> backend_spmd_benchmark(backend, 128, 2) end}
      ]
    end

    if Enum.empty?(benchmark_cases) do
      IO.puts("No non-CPU backends available for benchmarking")
      %{}
    else
      Benchee.run(
        benchmark_cases,
        time: 5,
        memory_time: 2,
        formatters: [
          Benchee.Formatters.Console
        ]
      )
    end
  end

  def run_torchx_scaling do
    Nx.default_backend(Torchx.Backend)

    Benchee.run(
      %{
        "TorchX SPMD (batch=64, 1 shard)" => fn -> torchx_spmd_benchmark(64, 1) end,
        "TorchX SPMD (batch=64, 2 shards)" => fn -> torchx_spmd_benchmark(64, 2) end,
        "TorchX SPMD (batch=64, 4 shards)" => fn -> torchx_spmd_benchmark(64, 4) end,
        "TorchX SPMD (batch=128, 1 shard)" => fn -> torchx_spmd_benchmark(128, 1) end,
        "TorchX SPMD (batch=128, 2 shards)" => fn -> torchx_spmd_benchmark(128, 2) end,
        "TorchX SPMD (batch=128, 4 shards)" => fn -> torchx_spmd_benchmark(128, 4) end
      },
      time: 5,
      memory_time: 2,
      formatters: [
        Benchee.Formatters.Console
      ]
    )
  end

  def run_torchx_mesh_scaling do
    # Configure TorchX for 12-core threading
    System.put_env("OMP_NUM_THREADS", "12")
    Nx.default_backend({Torchx.Backend, device: :cpu})

    # Test key mesh topologies: best performers, degenerate cases, and worst performers
    mesh_configs = [
      {1, 4},   # Best performer (degenerate)
      {4, 3},   # Good rectangular performer
      {12, 1},  # Linear degenerate (good performer)
      {1, 1},   # Single device degenerate (baseline)
      {1, 12},  # Worst degenerate case
      {2, 6}    # Previously thought optimal but underperforms
    ]

    benchmark_cases = for {x, y} = mesh <- mesh_configs do
      total_devices = x * y
      [
        {"TorchX SPMD mesh{#{x},#{y}} (batch=768, #{total_devices} devices)",
         fn -> torchx_mesh_benchmark(768, mesh) end},
        {"TorchX SPMD mesh{#{x},#{y}} (batch=1536, #{total_devices} devices)",
         fn -> torchx_mesh_benchmark(1536, mesh) end}
      ]
    end |> List.flatten() |> Enum.into(%{})

    Benchee.run(
      benchmark_cases,
      time: 2,
      memory_time: 1,
      formatters: [Benchee.Formatters.Console]
    )
  end

  def run_torchx_memory_scaling do
    # Configure TorchX for 12-core threading
    System.put_env("OMP_NUM_THREADS", "12")
    Nx.default_backend({Torchx.Backend, device: :cpu})

    # Test memory scaling with different batch sizes and core utilization
    Benchee.run(
      %{
        "TorchX Memory (batch=256, 1 core)" => fn -> torchx_memory_benchmark(256, {1, 1}) end,
        "TorchX Memory (batch=256, 4 cores)" => fn -> torchx_memory_benchmark(256, {4, 1}) end,
        "TorchX Memory (batch=256, 12 cores)" => fn -> torchx_memory_benchmark(256, {12, 1}) end
      },
      time: 2,
      memory_time: 1,
      formatters: [Benchee.Formatters.Console]
    )
  end

  def mps_spmd_benchmark(batch_size, num_shards) do
    feature_dim = 8
    sharding = AriaMath.Sharding.annotation({num_shards, 1}, [0, -1])

    # Create tensor with TorchX backend and move to MPS
    tensor = Nx.iota({batch_size, feature_dim}, backend: Torchx.Backend)
    mps_tensor = Torchx.to_device(tensor, "mps:0")

    # Apply SPMD operations on MPS
    sharded = AriaMath.Sharding.shard(mps_tensor, sharding)
    reduced = AriaMath.Sharding.all_reduce(sharded)
    gathered = AriaMath.Sharding.all_gather(reduced)

    gathered
  end

  # Benchmark implementations

  def spmd_scaling_benchmark(num_devices) do
    # Fixed total workload, distributed across devices
    total_batch_size = 64
    batch_per_device = div(total_batch_size, num_devices)
    feature_dim = 16

    # Simulate distributed SPMD execution
    tasks = for device_id <- 0..(num_devices-1) do
      Task.async(fn ->
        # Each device processes its shard
        start_idx = device_id * batch_per_device
        device_data = Nx.iota({batch_per_device, feature_dim}) |> Nx.add(start_idx * feature_dim)

        # Simulate SPMD computation (3D transformation pipeline)
        result = device_data
                 |> Nx.multiply(2.0)  # Scale
                 |> Nx.add(1.0)       # Translate
                 |> Nx.sin()          # Non-linear transformation

        result
      end)
    end

    # Wait for all devices and collect results
    results = Task.await_many(tasks, 5000)

    # Simulate AllReduce (sum all results)
    Enum.reduce(results, fn tensor, acc -> Nx.add(tensor, acc) end)
  end

  def memory_scaling_benchmark(batch_size, num_devices) do
    feature_dim = 8
    shard_size = div(batch_size, num_devices)

    # Create large tensor
    tensor = Nx.iota({batch_size, feature_dim}, backend: Torchx.Backend)

    # SPMD processing: simulate distributed computation
    device_results = for device <- 0..(num_devices-1) do
      start_idx = device * shard_size
      device_tensor = Nx.slice_along_axis(tensor, start_idx, shard_size, axis: 0)

      # Simulate computation on this shard
      device_tensor
      |> Nx.multiply(2.0)
      |> Nx.sin()
      |> Nx.cos()
    end

    # AllGather: concatenate results
    Nx.concatenate(device_results, axis: 0)
  end

  def torchx_spmd_benchmark(batch_size, num_shards) do
    feature_dim = 8
    sharding = AriaMath.Sharding.annotation({num_shards, 1}, [0, -1])

    # Create tensor with TorchX backend
    tensor = Nx.iota({batch_size, feature_dim}, backend: Torchx.Backend)

    # Apply SPMD operations
    sharded = AriaMath.Sharding.shard(tensor, sharding)
    reduced = AriaMath.Sharding.all_reduce(sharded)
    gathered = AriaMath.Sharding.all_gather(reduced)

    gathered
  end

  def backend_spmd_benchmark(backend, batch_size, num_shards) do
    feature_dim = 8
    sharding = AriaMath.Sharding.annotation({num_shards, 1}, [0, -1])

    # Create tensor with the specified backend
    tensor = AriaMath.BackendDetector.create_tensor(
      Nx.iota({batch_size, feature_dim}) |> Nx.to_list(),
      backend
    )

    # Apply SPMD operations
    sharded = AriaMath.Sharding.shard(tensor, sharding)
    reduced = AriaMath.Sharding.all_reduce(sharded)
    gathered = AriaMath.Sharding.all_gather(reduced)

    gathered
  end

  def torchx_mesh_benchmark(batch_size, {x, y}) do
    feature_dim = 8
    total_devices = x * y

    # Create sharding specification for the mesh
    sharding = AriaMath.Sharding.annotation({x, y}, [0, -1])

    # Create tensor with TorchX backend configured for threading
    tensor = Nx.iota({batch_size, feature_dim}, backend: {Torchx.Backend, device: :cpu})

    # Apply SPMD operations with mesh topology
    sharded = AriaMath.Sharding.shard(tensor, sharding)
    reduced = AriaMath.Sharding.all_reduce(sharded)
    gathered = AriaMath.Sharding.all_gather(reduced)

    gathered
  end

  def torchx_memory_benchmark(batch_size, {cores, _}) do
    feature_dim = 8
    shard_size = div(batch_size, cores)

    # Create large tensor with TorchX backend
    tensor = Nx.iota({batch_size, feature_dim}, backend: {Torchx.Backend, device: :cpu})

    # SPMD processing: simulate distributed computation across cores
    device_results = for core <- 0..(cores-1) do
      start_idx = core * shard_size
      end_idx = if core == cores-1 do
        batch_size - 1
      else
        start_idx + shard_size - 1
      end

      core_tensor = Nx.slice_along_axis(tensor, start_idx, shard_size, axis: 0)

      # Simulate memory-intensive computation on this core
      core_tensor
      |> Nx.multiply(2.0)
      |> Nx.sin()
      |> Nx.cos()
      |> Nx.exp()
    end

    # AllGather: concatenate results back together
    Nx.concatenate(device_results, axis: 0)
  end
end

# Run benchmarks if this file is executed directly
if __ENV__.file == __ENV__.file do
  IO.puts("Running TorchX SPMD Scaling Benchmarks (12-core Mac)...")
  IO.puts("======================================================")

  IO.puts("\n1. TorchX SPMD scaling with different mesh topologies:")
  AriaMath.SPMDScalingBench.run_torchx_mesh_scaling()

  IO.puts("\n2. Memory scaling with TorchX:")
  AriaMath.SPMDScalingBench.run_torchx_memory_scaling()

  IO.puts("\nBenchmarking complete! 🎯")
end
