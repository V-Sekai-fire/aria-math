# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.MixProject do
  use Mix.Project

  def project do
    [
      app: :aria_math,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs(),
      source_url: "https://github.com/V-Sekai-fire/aria-math"
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.10.0"},
      {:torchx, "~> 0.10"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end

  defp description do
    """
    Mathematical operations library for 3D graphics, including vectors, quaternions, 
    matrices, and geometric primitives with GPU acceleration support via Nx and TorchX.
    """
  end

  defp package do
    [
      name: "aria_math",
      files: ~w(lib mix.exs README.md),
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/V-Sekai-fire/aria-math"}
    ]
  end

  defp docs do
    [
      main: "AriaMath",
      source_url: "https://github.com/V-Sekai-fire/aria-math",
      extras: ["README.md"]
    ]
  end
end
