def get_point_key(len_board: int, len_board_column: int, row: int, column: int) -> int:
    """
    返回矩阵索引的哈希键。

    参数:
    - len_board: 矩阵的行数。
    - len_board_column: 矩阵的列数。
    - row: 当前行索引。
    - column: 当前列索引。

    返回:
    - int: 计算出的哈希键。

    示例:
    >>> get_point_key(10, 20, 1, 0)
    200
    """
    return len_board * len_board_column * row + column


def exits_word(
    board: list[list[str]],
    word: str,
    row: int,
    column: int,
    word_index: int,
    visited_points_set: set[int],
) -> bool:
    """
    从给定的起始位置检查是否可以找到单词的后缀。

    参数:
    - board: 字符矩阵。
    - word: 要查找的单词。
    - row: 当前行索引。
    - column: 当前列索引。
    - word_index: 当前单词的索引。
    - visited_points_set: 已访问点的集合。

    返回:
    - bool: 如果可以找到单词的后缀返回True，否则返回False。

    示例:
    >>> exits_word([["A"]], "B", 0, 0, 0, set())
    False
    """
    # 如果当前字符不匹配单词中的字符，返回False
    if board[row][column] != word[word_index]:
        return False

    # 如果已找到单词的最后一个字符，返回True
    if word_index == len(word) - 1:
        return True

    # 定义移动方向：右、左、上、下
    traverts_directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    len_board = len(board)
    len_board_column = len(board[0])

    # 遍历所有可能的方向
    for direction in traverts_directions:
        next_i = row + direction[0]
        next_j = column + direction[1]

        # 检查下一个位置是否在边界内
        if not (0 <= next_i < len_board and 0 <= next_j < len_board_column):
            continue

        # 获取下一个位置的哈希键
        key = get_point_key(len_board, len_board_column, next_i, next_j)
        # 如果该位置已被访问，跳过
        if key in visited_points_set:
            continue

        # 标记当前点为已访问
        visited_points_set.add(key)
        # 递归检查下一个字符
        if exits_word(board, word, next_i, next_j, word_index + 1, visited_points_set):
            return True

        # 回溯：移除当前点的访问标记
        visited_points_set.remove(key)

    return False  # 如果没有找到有效路径，返回False


def word_exists(board: list[list[str]], word: str) -> bool:
    """
    检查给定的单词是否存在于字符矩阵中。

    参数:
    - board: 字符矩阵。
    - word: 要查找的单词。

    返回:
    - bool: 如果单词存在返回True，否则返回False。

    示例:
    >>> word_exists([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED")
    True
    >>> word_exists([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE")
    True
    >>> word_exists([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB")
    False
    >>> word_exists([["A"]], "A")
    True
    >>> word_exists([["B", "A", "A"], ["A", "A", "A"], ["A", "B", "A"]], "ABB")
    False
    >>> word_exists([["A"]], 123)
    Traceback (most recent call last):
        ...
    ValueError: The word parameter should be a string of length greater than 0.
    >>> word_exists([["A"]], "")
    Traceback (most recent call last):
        ...
    ValueError: The word parameter should be a string of length greater than 0.
    >>> word_exists([[]], "AB")
    Traceback (most recent call last):
        ...
    ValueError: The board should be a non empty matrix of single chars strings.
    >>> word_exists([], "AB")
    Traceback (most recent call last):
        ...
    ValueError: The board should be a non empty matrix of single chars strings.
    >>> word_exists([["A"], [21]], "AB")
    Traceback (most recent call last):
        ...
    ValueError: The board should be a non empty matrix of single chars strings.
    """
    # 验证矩阵
    board_error_message = (
        "The board should be a non empty matrix of single chars strings."
    )

    len_board = len(board)
    if not isinstance(board, list) or len(board) == 0:
        raise ValueError(board_error_message)

    for row in board:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError(board_error_message)

        for item in row:
            if not isinstance(item, str) or len(item) != 1:
                raise ValueError(board_error_message)

    # 验证单词
    if not isinstance(word, str) or len(word) == 0:
        raise ValueError(
            "The word parameter should be a string of length greater than 0."
        )

    len_board_column = len(board[0])
    # 遍历矩阵的每个点作为起始点
    for i in range(len_board):
        for j in range(len_board_column):
            if exits_word(
                board, word, i, j, 0, {get_point_key(len_board, len_board_column, i, j)}
            ):
                return True  # 找到有效路径，返回True

    return False  # 如果没有找到有效路径，返回False


if __name__ == "__main__":
    import doctest

    doctest.testmod()  # 运行文档测试，验证函数的正确性
