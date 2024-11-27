def backtrack(input_string: str, word_dict: set[str], start: int) -> bool:
    """
    辅助函数，使用回溯法确定从索引'start'开始是否可以进行有效的单词拆分。

    参数:
    - input_string (str): 需要拆分的输入字符串。
    - word_dict (set[str]): 有效单词的集合。
    - start (int): 要检查的子字符串的起始索引。

    返回:
    - bool: 如果可以进行有效的拆分返回True，否则返回False。

    示例:
    >>> backtrack("leetcode", {"leet", "code"}, 0)
    True

    >>> backtrack("applepenapple", {"apple", "pen"}, 0)
    True

    >>> backtrack("catsandog", {"cats", "dog", "sand", "and", "cat"}, 0)
    False
    """

    # 基本情况：如果起始索引已到达字符串末尾
    if start == len(input_string):
        return True

    # 尝试从'start'到'end'的每一个可能的子字符串
    for end in range(start + 1, len(input_string) + 1):
        # 如果当前子字符串在字典中，并且后续部分也可以拆分
        if input_string[start:end] in word_dict and backtrack(
            input_string, word_dict, end
        ):
            return True  # 找到有效的拆分，返回True

    return False  # 如果没有找到有效的拆分，返回False


def word_break(input_string: str, word_dict: set[str]) -> bool:
    """
    确定输入字符串是否可以拆分为有效字典单词的序列。

    参数:
    - input_string (str): 需要拆分的输入字符串。
    - word_dict (set[str]): 有效单词的集合。

    返回:
    - bool: 如果字符串可以拆分为有效单词，返回True，否则返回False。

    示例:
    >>> word_break("leetcode", {"leet", "code"})
    True

    >>> word_break("applepenapple", {"apple", "pen"})
    True

    >>> word_break("catsandog", {"cats", "dog", "sand", "and", "cat"})
    False
    """

    # 调用辅助函数进行回溯检查
    return backtrack(input_string, word_dict, 0)


if __name__ == "__main__":
    # 示例用法
    print(word_break("leetcode", {"leet", "code"}))  # 输出: True
    print(word_break("applepenapple", {"apple", "pen"}))  # 输出: True
    print(word_break("catsandog", {"cats", "dog", "sand", "and", "cat"}))  # 输出: False
