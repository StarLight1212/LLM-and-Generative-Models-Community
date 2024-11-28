from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class BookNode:
    """书籍节点"""
    title: str
    is_borrowed: bool = False
    next_node: BookNode | None = None


@dataclass
class CircularLinkedList:
    """循环链表类"""
    head: BookNode | None = None  # 指向头节点
    tail: BookNode | None = None  # 指向尾节点

    def __iter__(self) -> Iterator[str]:
        """迭代书籍节点"""
        node = self.head
        if node is not None:
            while True:
                yield node.title
                node = node.next_node
                if node == self.head:
                    break

    def insert_tail(self, title: str) -> None:
        """在链表尾部插入新书籍"""
        new_node = BookNode(title)
        if self.head is None:  # 空链表
            new_node.next_node = new_node  # 第一个节点指向自己
            self.head = self.tail = new_node
        else:
            self.tail.next_node = new_node
            new_node.next_node = self.head
            self.tail = new_node

    def delete(self, title: str) -> bool:
        """删除指定书籍"""
        if self.head is None:
            return False

        current = self.head
        previous = self.tail

        while True:
            if current.title == title:
                if current == self.head:  # 删除头节点
                    if self.head == self.tail:  # 只有一个节点
                        self.head = self.tail = None
                    else:
                        previous.next_node = current.next_node
                        self.head = current.next_node
                else:  # 删除中间或尾节点
                    previous.next_node = current.next_node
                    if current == self.tail:  # 删除尾节点
                        self.tail = previous
                return True
            previous = current
            current = current.next_node
            if current == self.head:
                break
        return False

    def borrow_book(self, title: str) -> bool:
        """借阅书籍"""
        current = self.head
        if current is None:
            return False
        while True:
            if current.title == title:
                if not current.is_borrowed:
                    current.is_borrowed = True
                    return True
                else:
                    return False  # 书籍已被借出
            current = current.next_node
            if current == self.head:
                break
        return False  # 找不到书籍

    def return_book(self, title: str) -> bool:
        """归还书籍"""
        current = self.head
        if current is None:
            return False
        while True:
            if current.title == title:
                if current.is_borrowed:
                    current.is_borrowed = False
                    return True
                else:
                    return False  # 书籍未被借出
            current = current.next_node
            if current == self.head:
                break
        return False  # 找不到书籍

    def display_books(self) -> None:
        """显示所有书籍及其状态"""
        if self.head is None:
            print("没有书籍可显示。")
            return
        current = self.head
        print("书籍列表:")
        while True:
            status = "已借出" if current.is_borrowed else "可借出"
            print(f"{current.title} - {status}")
            current = current.next_node
            if current == self.head:
                break


# 实战应用示例：图书馆管理系统
if __name__ == "__main__":
    library = CircularLinkedList()

    # 添加书籍
    library.insert_tail("《活着》")
    library.insert_tail("《百年孤独》")
    library.insert_tail("《三体》")
    library.insert_tail("《小王子》")

    # 显示书籍
    library.display_books()

    # 借阅书籍
    print("\n借阅《三体》:")
    if library.borrow_book("《三体》"):
        print("借阅成功！")
    else:
        print("借阅失败，书籍已被借出。")

    # 显示书籍状态
    library.display_books()

    # 归还书籍
    print("\n归还《三体》:")
    if library.return_book("《三体》"):
        print("归还成功！")
    else:
        print("归还失败，书籍未被借出。")

    # 显示书籍状态
    library.display_books()

    # 删除书籍
    print("\n删除《小王子》:")
    if library.delete("《小王子》"):
        print("删除成功！")
    else:
        print("删除失败，书籍未找到。")

    # 显示书籍状态
    library.display_books()
