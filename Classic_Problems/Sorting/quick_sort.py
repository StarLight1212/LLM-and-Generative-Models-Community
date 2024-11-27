def quick_sort(arr):
    """
    快速排序函数
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
    |   选择基准 pivot    |
    |   pivot = arr[len(arr) // 2] |
    +---------------------+
              |
              v
    +---------------------+
    |   分割列表:        |
    |   left = [x for x in arr if x < pivot] |
    |   middle = [x for x in arr if x == pivot] |
    |   right = [x for x in arr if x > pivot] |
    +---------------------+
              |
              v
    +---------------------+
    |   递归排序并合并   |
    |   return quick_sort(left) + middle + quick_sort(right) |
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

    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准
    left = [x for x in arr if x < pivot]  # 小于基准的元素
    middle = [x for x in arr if x == pivot]  # 等于基准的元素
    right = [x for x in arr if x > pivot]  # 大于基准的元素

    # 递归排序并合并结果
    return quick_sort(left) + middle + quick_sort(right)

# 测试快速排序
if __name__ == "__main__":
    sample_list = [38, 27, 43, 3, 9, 82, 10]  # 待排序的列表
    print("原始列表:", sample_list)
    sorted_list = quick_sort(sample_list)  # 调用快速排序函数
    print("排序后的列表:", sorted_list)
