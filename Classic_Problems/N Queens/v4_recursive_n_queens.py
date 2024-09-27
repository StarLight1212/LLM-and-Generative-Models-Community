def solve_n_queens_recursive(n):
    def is_safe(board, row, col):
        # Check this column on upper side
        for i in range(row):
            if board[i] == col or \
                    board[i] - i == col - row or \
                    board[i] + i == col + row:
                return False
        return True

    def solve(board, row):
        if row == n:
            solutions.append(board[:])
            return

        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(board, row + 1)

    solutions = []
    board = [-1] * n  # -1 means no queen is placed in that row
    solve(board, 0)
    return solutions


# Example usage
n = 8
solutions = solve_n_queens_recursive(n)
print(f"Total solutions for {n} queens: {len(solutions)}")
for solution in solutions:
    print(solution)
