# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.AxonLayers.WorkflowPlanner.Execution do
  @moduledoc """
  Execution functions for WorkflowPlanner.

  This module contains functions responsible for plan execution,
  including todo item processing and blacklisting.
  """

  @doc """
  Get the todo items from the solution graph.
  """
  @spec extract_todo_items(any()) :: [tuple()]
  def extract_todo_items(_solution_graph) do
    # Implementation to extract todo items from the solution graph
    # This would convert the graph structure back to executable todo items
    raise "TODO: Implement extract_todo_items"
  end

  @doc """
  Execute a list of todo items (actions, unigoals, etc.)
  """
  @spec execute_todo_items([tuple()], map(), map(), map()) :: {map(), map()}
  def execute_todo_items(todo_items, initial_state, initial_blacklist, domain) do
    Enum.reduce(todo_items, {initial_state, initial_blacklist}, fn
      todo_item, {acc_state, acc_blacklist} ->
        execute_single_todo_item(todo_item, acc_state, acc_blacklist, domain)
    end)
  end

  @doc """
  Execute a single todo item.
  """
  @spec execute_single_todo_item(tuple(), map(), map(), map()) :: {map(), map()}
  def execute_single_todo_item(todo_item, state, blacklist, domain) do
    case todo_item do
      # Primitive action
      {action_name, action_args} when is_atom(action_name) ->
        if is_map(domain.actions) && Map.has_key?(domain.actions, action_name) do
          # Check if action is blacklisted
          if AriaPlanner.Planner.Blacklisting.command_blacklisted?(blacklist, {action_name, action_args}) do
            raise "Action blacklisted: #{action_name}"
          end

          case domain.actions[action_name].(state, action_args) do
            {:ok, new_state, _metadata} ->
              {new_state, blacklist}

            _ ->
              # Blacklist the failed action and raise error
              raise "Action execution failed: #{action_name}"
          end
        else
          # Skip unknown actions
          {state, blacklist}
        end

      # Unigoal call
      {:unigoal, unigoal_name, unigoal_args} ->
        # Check if unigoal method is blacklisted
        if AriaPlanner.Planner.Blacklisting.method_blacklisted?(blacklist, unigoal_name) do
          raise "Unigoal method blacklisted: #{unigoal_name}"
        end

        case domain.unigoal_methods[unigoal_name].(state, unigoal_args) do
          {:ok, sub_todo_items, _metadata} ->
            # Recursively execute the unigoal's todo items
            execute_todo_items(sub_todo_items, state, blacklist, domain)

          _ ->
            # Blacklist the failed unigoal method and raise error
            raise "Unigoal task execution failed: #{unigoal_name}"
        end

      # Method call (for decomposition)
      {:method, method_name, method_args} ->
        if is_map(domain.methods) && Map.has_key?(domain.methods, method_name) do
          # Check if method is blacklisted
          if AriaPlanner.Planner.Blacklisting.method_blacklisted?(blacklist, Atom.to_string(method_name)) do
            raise "Method blacklisted: #{method_name}"
          end

          case domain.methods[method_name].(state, method_args) do
            {:ok, sub_todo_items, _metadata} ->
              # Recursively execute the method's todo items
              execute_todo_items(sub_todo_items, state, blacklist, domain)

            _ ->
              # Blacklist the failed method and raise error
              raise "Method execution failed: #{method_name}"
          end
        else
          # Skip unknown methods
          {state, blacklist}
        end

      # Skip unknown todo item types
      _ ->
        {state, blacklist}
    end
  end
end
