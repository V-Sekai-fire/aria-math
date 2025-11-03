# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.GoalSolver.Model do
  @moduledoc """
  Axon model building for goal solving.

  This module contains the neural network model definitions
  for goal achievement prediction and solution generation.
  """



  @doc """
  Build an Axon model for goal solving.

  ## Parameters
  - `opts` - Model configuration options
    - `:max_goals` - Maximum number of goals (default: 10)
    - `:embedding_size` - Size of goal embeddings (default: 32)

  ## Returns
  - Axon model that can process goals and state to predict solutions

  ## Usage
  ```elixir
  model = GoalSolver.Model.build_model(max_goals: 5)
  params = Axon.init(model)
  # Use for predicting goal solutions
  ```
  """
  @spec build_model(keyword()) :: Axon.t()
  def build_model(opts \\ []) do
    max_goals = Keyword.get(opts, :max_goals, 10)
    embedding_size = Keyword.get(opts, :embedding_size, 32)

    # Input: goal representations (embeddings)
    # Shape: [batch_size, max_goals, embedding_size]
    goals_input = Axon.input("goals", shape: {nil, max_goals, embedding_size})

    # Input: state representation
    # Shape: [batch_size, state_size]
    state_input = Axon.input("state", shape: {nil, embedding_size})

    # Goal encoder - process individual goals
    goal_encoded = goals_input
                   |> Axon.dense(64, name: "goal_encoder1")
                   |> Axon.relu()
                   |> Axon.dense(32, name: "goal_encoder2")

    # State encoder - process state
    state_encoded = state_input
                    |> Axon.dense(64, name: "state_encoder1")
                    |> Axon.relu()
                    |> Axon.dense(32, name: "state_encoder2")

    # Expand state to match goals shape for attention
    expanded_state = state_encoded
                     |> Axon.expand_dims(axes: [1])
                     |> Axon.broadcast(Axon.shape(goal_encoded))

    # Attention mechanism - how goals relate to state
    attention_input = Axon.concatenate([goal_encoded, expanded_state])
    attention = attention_input
                |> Axon.dense(16, name: "attention")
                |> Axon.sigmoid()

    # Goal achievement predictor
    achievement_pred = attention
                       |> Axon.dense(1, name: "achievement_pred")
                       |> Axon.sigmoid()
                       |> Axon.squeeze(axes: [2])

    # Solution assignment predictor (simplified)
    solution_pred = attention_input
                    |> Axon.dense(8, name: "solution_pred")
                    |> Axon.tanh()

    # Combine outputs
    Axon.container(%{
      goal_achievement: achievement_pred,
      solution_assignments: solution_pred
    })
  end
end
