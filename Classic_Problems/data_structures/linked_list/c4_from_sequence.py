"""
递归程序：从序列创建链表并打印其字符串表示。

链表是一种动态数据结构，适合需要频繁插入和删除的场景。在本示例中，我们将实现一个简单的链表，并结合一个实战应用：管理一个学生的成绩单。
"""

class Node:
    """链表节点"""
    def __init__(self, data=None):
        self.data = data
        self.next = None

    def __repr__(self):
        """返回节点及其后续节点的可视化表示"""
        string_rep = ""
        temp = self
        while temp:
            string_rep += f"<{temp.data}> ---> "
            temp = temp.next
        string_rep += "<END>"
        return string_rep


def make_linked_list(elements_list):
    """从给定序列（列表/元组）创建链表并返回链表头"""
    if not elements_list:
        raise Exception("元素列表为空")

    head = Node(elements_list[0])  # 设置第一个元素为头节点
    current = head
    for data in elements_list[1:]:  # 遍历剩余元素
        current.next = Node(data)
        current = current.next
    return head


# 实战应用示例：管理学生成绩单
if __name__ == "__main__":
    """
    本示例展示如何使用链表管理学生的成绩单。
    """
    grades = [85, 90, 78, 92, 88]  # 学生成绩列表
    print(f"成绩列表: {grades}")
    print("正在创建链表...")
    linked_list = make_linked_list(grades)
    print("成绩单链表:")
    print(linked_list)
