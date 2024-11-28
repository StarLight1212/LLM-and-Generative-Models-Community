"""
链表是一种动态数据结构，由一系列节点组成，每个节点包含数据和指向下一个节点的指针。链表的优点在于可以灵活地插入和删除节点，适合需要频繁修改的场景。

在本示例中，我们将实现一个简单的链表，并结合一个实战应用：管理一个购物清单。
"""

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    """链表节点"""
    data: Any
    next_node: Node | None = None

    def __repr__(self) -> str:
        """返回节点的字符串表示"""
        return f"Node({self.data})"


class LinkedList:
    """链表类"""
    def __init__(self):
        """初始化链表"""
        self.head = None

    def __iter__(self) -> Iterator[Any]:
        """返回链表的迭代器"""
        node = self.head
        while node:
            yield node.data
            node = node.next_node

    def __len__(self) -> int:
        """返回链表的长度"""
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        """返回链表的字符串表示"""
        return " -> ".join([str(item) for item in self])

    def insert_tail(self, data: Any) -> None:
        """在链表尾部插入数据"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next_node:
            current = current.next_node
        current.next_node = new_node

    def delete_head(self) -> Any:
        """删除头节点并返回其数据"""
        if not self.head:
            raise IndexError("链表为空，无法删除头节点。")
        data = self.head.data
        self.head = self.head.next_node
        return data

    def delete_tail(self) -> Any:
        """删除尾节点并返回其数据"""
        if not self.head:
            raise IndexError("链表为空，无法删除尾节点。")
        if not self.head.next_node:
            data = self.head.data
            self.head = None
            return data
        current = self.head
        while current.next_node and current.next_node.next_node:
            current = current.next_node
        data = current.next_node.data
        current.next_node = None
        return data

    def print_list(self) -> None:
        """打印链表中的所有节点数据"""
        print(self)


# 实战应用示例：管理购物清单
if __name__ == "__main__":
    """
    本示例展示如何使用链表管理购物清单。
    """
    shopping_list = LinkedList()
    
    # 添加购物项
    shopping_list.insert_tail("牛奶")
    shopping_list.insert_tail("面包")
    shopping_list.insert_tail("鸡蛋")
    
    print("当前购物清单:")
    shopping_list.print_list()
    
    # 删除购物项
    print("删除头部购物项:", shopping_list.delete_head())
    print("删除尾部购物项:", shopping_list.delete_tail())
    
    print("更新后的购物清单:")
    shopping_list.print_list()
