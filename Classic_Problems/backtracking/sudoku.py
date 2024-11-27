from __future__ import annotations

Matrix = list[list[int]]

# 初始化数独网格
initial_grid: Matrix = [
    [3, 0, 6, 5, 0, 8, 4, 0, 0],
    [5, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 7, 0, 0, 0, 0, 3, 1],
    [0, 0, 3, 0, 1, 0, 0, 8, 0],
    [9, 0, 0, 8, 6, 3, 0, 0, 5],
    [0, 5, 0, 0, 9, 0, 6, 0, 0],
    [1, 3, 0, 0, 0, 0, 2, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 4],
    [0, 0, 5, 2, 0, 6, 3, 0, 0],
]

# 一个无解的数独网格
no_solution: Matrix = [
    [5, 0, 6, 5, 0, 8, 4, 0, 3],
    [5, 2, 0, 0, 0, 0, 0, 0, 2],
    [1, 8, 7, 0, 0, 0, 0, 3, 1],
    [0, 0, 3, 0, 1, 0, 0, 8, 0],
    [9, 0, 0, 8, 6, 3, 0, 0, 5],
    [0, 5, 0, 0, 9, 0, 6, 0, 0],
    [1, 3, 0, 0, 0, 0, 2, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 4],
    [0, 0, 5, 2, 0, 6, 3, 0, 0],
]

def is_safe(grid: Matrix, row: int, column: int, n: int) -> bool:
    """
    检查在给定的行、列和3x3子网格中是否可以安全地放置数字n。

    参数:
    - grid: 当前的数独网格。
    - row: 当前行索引。
    - column: 当前列索引。
    - n: 要放置的数字。

    返回:
    - bool: 如果可以安全放置返回True，否则返回False。
    """
    # 检查行和列
    for i in range(9):
        if n in {grid[row][i], grid[i][column]}:
            return False

    # 检查3x3子网格
    start_row, start_col = 3 * (row // 3), 3 * (column // 3)
    for i in range(3):
        for j in range(3):
            if grid[start_row + i][start_col + j] == n:
                return False

    return True  # 如果没有冲突，返回True


def find_empty_location(grid: Matrix) -> tuple[int, int] | None:
    """
    查找网格中的空位置（值为0的单元格）。

    参数:
    - grid: 当前的数独网格。

    返回:
    - tuple[int, int] | None: 如果找到空位置，返回其坐标；否则返回None。
    """
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return i, j
    return None  # 如果没有找到空位置，返回None


def sudoku(grid: Matrix) -> Matrix | None:
    """
    尝试填充数独网格，返回解决后的网格或None（如果无解）。

    参数:
    - grid: 当前的数独网格。

    返回:
    - Matrix | None: 如果找到解决方案，返回填充后的网格；否则返回None。

    示例:
    >>> sudoku(initial_grid)  # doctest: +NORMALIZE_WHITESPACE
    [[3, 1, 6, 5, 7, 8, 4, 9, 2],
     [5, 2, 9, 1, 3, 4, 7, 6, 8],
     [4, 8, 7, 6, 2, 9, 5, 3, 1],
     [2, 6, 3, 4, 1, 5, 9, 8, 7],
     [9, 7, 4, 8, 6, 3, 1, 2, 5],
     [8, 5, 1, 7, 9, 2, 6, 4, 3],
     [1, 3, 8, 9, 4, 7, 2, 5, 6],
     [6, 9, 2, 3, 5, 1, 8, 7, 4],
     [7, 4, 5, 2, 8, 6, 3, 1, 9]]
    >>> sudoku(no_solution) is None
    True
    """
    # 查找空位置
    if location := find_empty_location(grid):
        row, column = location
    else:
        # 如果没有空位置，数独已解决
        return grid

    # 尝试填入1到9的数字
    for digit in range(1, 10):
        if is_safe(grid, row, column, digit):
            grid[row][column] = digit  # 填入数字

            # 递归调用，检查下一个位置
            if sudoku(grid) is not None:
                return grid  # 如果找到解决方案，返回网格

            grid[row][column] = 0  # 回溯，重置当前单元格

    return None  # 如果没有找到解决方案，返回None


def print_solution(grid: Matrix) -> None:
    """
    打印数独网格的解决方案。

    参数:
    - grid: 当前的数独网格。
    """
    for row in grid:
        print(" ".join(str(cell) for cell in row))


if __name__ == "__main__":
    # 测试数独求解器
    for example_grid in (initial_grid, no_solution):
        print("\nExample grid:\n" + "=" * 20)
        print_solution(example_grid)
        print("\nExample grid solution:")
        solution = sudoku(example_grid)
        if solution is not None:
            print_solution(solution)
        else:
            print("Cannot find a solution.")
