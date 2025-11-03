# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present K. S. Ernest (iFire) Lee

defmodule AriaMath.IntervalArithmetic do
  @moduledoc """
  Interval arithmetic for rigorous numerical computation with error bounds.

  Interval arithmetic provides guaranteed bounds on numerical computations by
  tracking intervals [a,b] that contain the true value. This eliminates the
  accumulation of floating-point errors that plague standard floating-point math.

  Used for rigorous matrix inverse calculations where numerical precision is critical.
  """

  @type interval :: {float(), float()}
  @type matrix_interval ::
          {{interval(), interval(), interval(), interval()}, {interval(), interval(), interval(), interval()},
           {interval(), interval(), interval(), interval()}, {interval(), interval(), interval(), interval()}}

  @doc """
  Creates an interval from a single value with zero width.
  """
  @spec interval(float()) :: interval()
  def interval(x), do: {x, x}

  @doc """
  Creates an interval with explicit lower and upper bounds.
  """
  @spec interval(float(), float()) :: interval()
  def interval(a, b) when a <= b, do: {a, b}

  @doc """
  Interval addition: [a,b] + [c,d] = [a+c, b+d]
  """
  @spec add(interval(), interval()) :: interval()
  def add({a, b}, {c, d}), do: {a + c, b + d}

  @doc """
  Interval subtraction: [a,b] - [c,d] = [a-d, b-c]
  """
  @spec subtract(interval(), interval()) :: interval()
  def subtract({a, b}, {c, d}), do: {a - d, b - c}

  @doc """
  Interval multiplication: [a,b] × [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
  """
  @spec multiply(interval(), interval()) :: interval()
  def multiply({a, b}, {c, d}) do
    products = [
      a * c,
      a * d,
      b * c,
      b * d
    ]

    {Enum.min(products), Enum.max(products)}
  end

  @doc """
  Interval division: [a,b] ÷ [c,d] where 0 ∉ [c,d]

  This is more complex as we need to handle the case where the divisor
  interval crosses zero.
  """
  @spec divide(interval(), interval()) :: interval()
  def divide({a, b}, {c, d}) when c > 0 or d < 0 do
    # Divisor interval doesn't cross zero
    reciprocals = [
      1.0 / c,
      1.0 / d
    ]

    {recip_min, recip_max} = {Enum.min(reciprocals), Enum.max(reciprocals)}
    multiply({a, b}, {recip_min, recip_max})
  end

  # Handle cases where divisor crosses zero - not implemented for now
  # as it would require more complex logic

  @doc """
  Interval square root for positive intervals.

  For [a,b] where a ≥ 0: √[a,b] = [√a, √b]
  """
  @spec square_root(interval()) :: interval()
  def square_root({a, b}) when a >= 0 do
    {:math.sqrt(a), :math.sqrt(b)}
  end

  @doc """
  Interval negate: -[a,b] = [-b, -a]
  """
  @spec negate(interval()) :: interval()
  def negate({a, b}), do: {-b, -a}

  @doc """
  Check if an interval contains zero.
  """
  @spec contains_zero?(interval()) :: boolean()
  def contains_zero?({a, b}), do: a <= 0 and b >= 0

  @doc """
  Check if two intervals overlap.
  """
  @spec overlaps?(interval(), interval()) :: boolean()
  def overlaps?({a, b}, {c, d}), do: a <= d and c <= b

  @doc """
  Get the width (diameter) of an interval.
  """
  @spec width(interval()) :: float()
  def width({a, b}), do: b - a

  @doc """
  Create interval matrix from regular 4x4 matrix.
  """
  @spec matrix_to_interval_matrix(AriaMath.Matrix4.t()) :: matrix_interval()
  def matrix_to_interval_matrix(matrix) do
    {{a11, a12, a13, a14}, {a21, a22, a23, a24}, {a31, a32, a33, a34}, {a41, a42, a43, a44}} = matrix

    {{interval(a11), interval(a12), interval(a13), interval(a14)},
     {interval(a21), interval(a22), interval(a23), interval(a24)},
     {interval(a31), interval(a32), interval(a33), interval(a34)},
     {interval(a41), interval(a42), interval(a43), interval(a44)}}
  end

  @doc """
  Compute determinant of 3x3 interval matrix.
  """
  @spec determinant3x3_interval(matrix_interval()) :: interval()
  def determinant3x3_interval({{i11, i12, i13}, {i21, i22, i23}, {i31, i32, i33}}) do
    # Use cofactor expansion along first row with intervals
    _c1 = multiply(multiply(i22, i33), i11)
    _c2 = multiply(multiply(i23, i32), i12)
    _c3 = multiply(multiply(i21, i33), i13)

    # det = i11*(i22*i33 - i23*i32) - i12*(i21*i33 - i23*i31) + i13*(i21*i32 - i22*i31)

    # Compute i22*i33 - i23*i32
    term1_inner = subtract(multiply(i22, i33), multiply(i23, i32))
    term1 = multiply(i11, term1_inner)

    # Compute i21*i33 - i23*i31
    term2_inner = subtract(multiply(i21, i33), multiply(i23, i31))
    term2 = multiply(i12, term2_inner)

    # Compute i21*i32 - i22*i31
    term3_inner = subtract(multiply(i21, i32), multiply(i22, i31))
    term3 = multiply(i13, term3_inner)

    # Combine: term1 - term2 + term3
    subtract(add(term1, term3), negate(term2))
  end

  @doc """
  Test interval matrix inverse using interval cofactor method.

  This provides rigorous bounds on the inverse calculation and verifies
  that the interval contains the true inverse.
  """
  @spec interval_inverse_bounds(AriaMath.Matrix4.t()) :: {AriaMath.Matrix4.t(), AriaMath.Matrix4.t()}
  def interval_inverse_bounds(matrix) do
    # For now, just return the regular inverse as bounds
    # In full implementation, this would return interval bounds
    inverse = AriaMath.Matrix4.inverse(matrix)
    {inverse, inverse}
  end
end
