from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    data: Any
    next_node: Node | None = None


@dataclass
class LinkedList:
    head: Node | None = None

    def __iter__(self) -> Iterator:
        """返回链表的迭代器"""
        node = self.head
        while node:
            yield node.data
            node = node.next_node

    def __len__(self) -> int:
        """返回链表的长度"""
        return sum(1 for _ in self)

    def push(self, new_data: Any) -> None:
        """在链表头部添加新节点"""
        new_node = Node(new_data)
        new_node.next_node = self.head
        self.head = new_node

    def swap_nodes(self, node_data_1: Any, node_data_2: Any) -> None:
        """
        交换链表中两个节点的位置

        Args:
            node_data_1: 第一个节点的数据值
            node_data_2: 第二个节点的数据值
        """
        if node_data_1 == node_data_2:
            return

        # 查找两个节点
        node_1, node_2 = self.head, self.head
        while node_1 and node_1.data != node_data_1:
            node_1 = node_1.next_node
        while node_2 and node_2.data != node_data_2:
            node_2 = node_2.next_node

        # 如果任一节点未找到，则不进行交换
        if node_1 is None or node_2 is None:
            return

        # 交换节点的数据值
        node_1.data, node_2.data = node_2.data, node_1.data


# 实战应用示例：管理一个待办事项列表
if __name__ == "__main__":
    """
    Python脚本，输出链表中节点的交换。
    """
    from doctest import testmod

    testmod()
    todo_list = LinkedList()
    # 添加待办事项
    for task in ["买菜", "洗衣服", "读书", "锻炼", "写代码"]:
        todo_list.push(task)

    print(f"原始待办事项列表: {list(todo_list)}")
    # 交换两个待办事项
    todo_list.swap_nodes("买菜", "写代码")
    print(f"修改后的待办事项列表: {list(todo_list)}")
    print("交换了待办事项 '买菜' 和 '写代码' 的位置。")
