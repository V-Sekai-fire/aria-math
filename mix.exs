# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.MixProject do
  use Mix.Project

  def project do
    [
      app: :aria_math,
      version: "0.1.0",
      build_embedded: true,
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: [
        main: "AriaMath",
        source_url: "https://github.com/V-Sekai-fire/aria-hybrid-planner",
        source_ref: "main",
        extras: ["README.md"],
        groups_for_modules: [
          "Differentiable Math": [
            "Gradient-enabled mathematical operations for AI learning",
            AriaMath.Vector3,
            AriaMath.Matrix4,
            AriaMath.Matrix4.Tensor
          ],
          "Axon Components": [
            "Neural network and tensor-based operations",
            AriaMath.AxonLayers.GoalSolver.Model,
            AriaMath.AxonLayers.StnSolver.Axon
          ],
          "Planning Core": [
            "Traditional planning algorithms and data structures",
            AriaMath.AxonLayers.Common,
            AriaMath.AxonLayers.SolutionTensorGraph,
            AriaMath.AxonLayers.GoalSolver,
            AriaMath.AxonLayers.StnSolver,
            AriaMath.AxonLayers.WorkflowPlanner
          ],
          "Goal Solving": [
            "Goal optimization and constraint satisfaction",
            AriaMath.AxonLayers.GoalSolver.Core,
            AriaMath.AxonLayers.GoalSolver.Utils
          ],
          "Temporal Reasoning": [
            "STN constraint solving and time management",
            AriaMath.AxonLayers.StnSolver.Core
          ],
          "Workflow Planning": [
            "Plan execution and workflow management",
            AriaMath.AxonLayers.WorkflowPlanner.Execution,
            AriaMath.AxonLayers.WorkflowPlanner.Solver,
            AriaMath.AxonLayers.WorkflowPlanner.GraphBuilder
          ],
          "Graph Operations": [
            "Solution graph construction and analysis",
            AriaMath.AxonLayers.SolutionTensorGraph.Builders,
            AriaMath.AxonLayers.SolutionTensorGraph.Extractors
          ]
        ]
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.10"},
      {:axon, "~> 0.7"},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.30", only: :dev, runtime: false},
    ]
  end
end
