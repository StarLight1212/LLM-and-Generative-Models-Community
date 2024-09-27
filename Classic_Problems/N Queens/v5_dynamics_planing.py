def solve_n_queens_dp(n):
    def place_queens(row, cols, diag1, diag2):
        if row == n:
            solutions.append(queens[:])
            return
        # Available positions
        available_positions = ((1 << n) - 1) & ~(cols | diag1 | diag2)
        while available_positions:
            # Get the rightmost bit that is set
            pos = available_positions & -available_positions
            available_positions ^= pos  # Remove this position from available
            column = bin(pos).count('0') - 1  # Get the column index
            queens[row] = column
            place_queens(row + 1,
                         cols | pos,
                         (diag1 | pos) << 1,
                         (diag2 | pos) >> 1)

    solutions = []
    queens = [-1] * n
    place_queens(0, 0, 0, 0)
    return solutions


# Example usage
n = 8
solutions = solve_n_queens_dp(n)
print(f"Total solutions for {n} queens: {len(solutions)}")
for solution in solutions:
    print(solution)
