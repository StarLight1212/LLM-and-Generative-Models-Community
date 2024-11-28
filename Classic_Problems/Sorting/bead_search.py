def bead_sort(sequence: list) -> list:
    """
    >>> bead_sort([6, 11, 12, 4, 1, 5])
    [1, 4, 5, 6, 11, 12]

    >>> bead_sort([9, 8, 7, 6, 5, 4 ,3, 2, 1])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> bead_sort([5, 0, 4, 3])
    [0, 3, 4, 5]

    >>> bead_sort([8, 2, 1])
    [1, 2, 8]

    >>> bead_sort([1, .9, 0.0, 0, -1, -.9])
    Traceback (most recent call last):
        ...
    TypeError: Sequence must be list of non-negative integers

    >>> bead_sort("Hello world")
    Traceback (most recent call last):
        ...
    TypeError: Sequence must be list of non-negative integers
    """
    # 检查输入序列是否为非负整数
    if any(not isinstance(x, int) or x < 0 for x in sequence):
        raise TypeError("Sequence must be list of non-negative integers")
    
    # 逻辑框图：
    # 1. 遍历序列的长度
    # 2. 比较相邻的两个元素
    # 3. 如果前一个元素大于后一个元素，进行调整
    # 4. 返回排序后的序列

    for _ in range(len(sequence)):
        for i, (rod_upper, rod_lower) in enumerate(zip(sequence, sequence[1:])):  # noqa: RUF007
            if rod_upper > rod_lower:
                sequence[i] -= rod_upper - rod_lower
                sequence[i + 1] += rod_upper - rod_lower
    return sequence

# 实战应用示例：对购物车中的商品价格进行排序
if __name__ == "__main__":
    # 假设我们有一个购物车，里面有商品的价格
    prices = [5, 4, 3, 2, 1]
    sorted_prices = bead_sort(prices)
    print("排序后的商品价格:", sorted_prices)  # 输出: [1, 2, 3, 4, 5]

    # 另一个示例
    assert bead_sort([7, 9, 4, 3, 5]) == [3, 4, 5, 7, 9]
