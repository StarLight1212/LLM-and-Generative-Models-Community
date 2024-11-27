import random

def randomized_quick_sort(arr):
    """
    随机快速排序函数
    :param arr: 待排序的列表
    :return: 排序后的列表
    +---------------------+
    |   开始              |
    +---------------------+
              |
              v
    +---------------------+
    |   输入待排序列表    |
    +---------------------+
              |
              v
    +---------------------+
    |   如果列表长度 <= 1 |
    |       返回列表      |
    +---------------------+
              |
              v
    +---------------------+
    |   随机选择基准 pivot |
    |   pivot_index = random.randint(0, len(arr) - 1) |
    |   pivot = arr[pivot_index] |
    +---------------------+
              |
              v
    +---------------------+
    |   将基准元素移到最后 |
    |   arr[pivot_index], arr[-1] = arr[-1], arr[pivot_index] |
    +---------------------+
              |
              v
    +---------------------+
    |   分割列表:        |
    |   left = [x for x in arr[:-1] if x < pivot] |
    |   middle = [x for x in arr if x == pivot] |
    |   right = [x for x in arr[:-1] if x > pivot] |
    +---------------------+
              |
              v
    +---------------------+
    |   递归排序并合并   |
    |   return randomized_quick_sort(left) + middle + randomized_quick_sort(right) |
    +---------------------+
              |
              v
    +---------------------+
    |   返回排序后的列表  |
    +---------------------+
              |
              v
    +---------------------+
    |   结束              |
    +---------------------+
    """
    if len(arr) <= 1:  # 如果列表长度小于等于1，直接返回
        return arr

    # 随机选择一个基准元素
    pivot_index = random.randint(0, len(arr) - 1)  # 随机选择基准的索引
    pivot = arr[pivot_index]  # 获取基准元素
    # 将基准元素移到列表的最后
    arr[pivot_index], arr[-1] = arr[-1], arr[pivot_index]

    # 分割列表
    left = [x for x in arr[:-1] if x < pivot]  # 小于基准的元素
    middle = [x for x in arr if x == pivot]  # 等于基准的元素
    right = [x for x in arr[:-1] if x > pivot]  # 大于基准的元素

    # 递归排序并合并结果
    return randomized_quick_sort(left) + middle + randomized_quick_sort(right)

# 测试随机快速排序
if __name__ == "__main__":
    sample_list = [38, 27, 43, 3, 9, 82, 10]  # 待排序的列表
    print("原始列表:", sample_list)
    sorted_list = randomized_quick_sort(sample_list)  # 调用随机快速排序函数
    print("排序后的列表:", sorted_list)
