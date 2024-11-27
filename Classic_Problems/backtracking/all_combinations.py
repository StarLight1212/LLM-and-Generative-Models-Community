from itertools import combinations

def combination_lists(n: int, k: int) -> list[list[int]]:
    """
    生成从1到n中选择k个数的所有组合
    :param n: 选择的最大数字
    :param k: 选择的数字个数
    :return: 所有可能的组合列表
    >>> combination_lists(n=4, k=2)
    [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    """
    # 使用itertools库中的combinations函数生成所有组合
    return [list(x) for x in combinations(range(1, n + 1), k)]

def generate_all_combinations(n: int, k: int) -> list[list[int]]:
    """
    生成从1到n中选择k个数的所有组合
    :param n: 选择的最大数字
    :param k: 选择的数字个数
    :return: 所有可能的组合列表
    """
    # 检查k和n的有效性
    if k < 0 or n < 0:
        raise ValueError("n and k must not be negative")  # k和n不能为负数

    result = []  # 用于存储最终的组合结果
    create_all_state(1, n, k, [], result)  # 调用递归函数生成组合
    return result  # 返回生成的组合列表

def create_all_state(increment: int, total_number: int, level: int, current_list: list[int], total_list: list[list[int]]) -> None:
    """
    递归生成所有组合的状态
    :param increment: 当前选择的起始数字
    :param total_number: 最大数字
    :param level: 剩余需要选择的数字个数
    :param current_list: 当前已选择的数字列表
    :param total_list: 存储所有组合的列表
    """
    if level == 0:  # 如果没有剩余需要选择的数字
        total_list.append(current_list[:])  # 将当前组合添加到结果列表中
        return

    # 从当前数字开始，遍历到可以选择的最大数字
    for i in range(increment, total_number - level + 2):
        current_list.append(i)  # 将当前数字添加到组合中
        # 递归调用，选择下一个数字，减少需要选择的数量
        create_all_state(i + 1, total_number, level - 1, current_list, total_list)
        current_list.pop()  # 回溯，移除最后添加的数字

if __name__ == "__main__":
    from doctest import testmod  # 导入doctest模块用于测试
    testmod()  # 运行文档测试，验证函数的正确性

    # 测试生成组合的功能
    print(generate_all_combinations(n=4, k=2))  # 输出从1到4中选择2个数的组合

    # 验证generate_all_combinations与combination_lists的结果是否一致
    for n in range(1, 5):
        for k in range(1, n + 1):
            print(n, k, generate_all_combinations(n, k) == combination_lists(n, k))

    print("Benchmark:")  # 基准测试
    from timeit import timeit  # 导入timeit模块用于性能测试
    for func in ("combination_lists", "generate_all_combinations"):
        # 测试两个函数的执行时间
        print(f"{func:>25}(): {timeit(f'{func}(n=4, k=2)', globals=globals())}")
