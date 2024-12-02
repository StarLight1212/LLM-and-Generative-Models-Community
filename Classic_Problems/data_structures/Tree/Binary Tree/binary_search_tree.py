from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Self


@dataclass
class Node:
    value: int
    left: Node | None = None
    right: Node | None = None
    parent: Node | None = None  # 方便删除节点

    def __iter__(self) -> Iterator[int]:
        """遍历节点"""
        yield from self.left or []
        yield self.value
        yield from self.right or []

    def __repr__(self) -> str:
        """节点的字符串表示"""
        return f"{self.value}" if not self.left and not self.right else f"{self.value}: ({self.left}, {self.right})"

    @property
    def is_right(self) -> bool:
        """判断当前节点是否为右子节点"""
        return self.parent and self is self.parent.right


@dataclass
class BinarySearchTree:
    root: Node | None = None

    def __bool__(self) -> bool:
        return self.root is not None

    def __iter__(self) -> Iterator[int]:
        """遍历树"""
        yield from self.root or []

    def __str__(self) -> str:
        """返回树的字符串表示"""
        return str(self.root)

    def empty(self) -> bool:
        """判断树是否为空"""
        return self.root is None

    def insert(self, *values) -> Self:
        """插入新节点"""
        for value in values:
            self._insert(value)
        return self

    def _insert(self, value: int) -> None:
        """插入单个节点"""
        new_node = Node(value)
        if self.empty():
            self.root = new_node
        else:
            parent_node = self.root
            while True:
                if value < parent_node.value:
                    if parent_node.left is None:
                        parent_node.left = new_node
                        new_node.parent = parent_node
                        break
                    parent_node = parent_node.left
                else:
                    if parent_node.right is None:
                        parent_node.right = new_node
                        new_node.parent = parent_node
                        break
                    parent_node = parent_node.right

    def search(self, value: int) -> Node | None:
        """查找节点"""
        if self.empty():
            raise IndexError("Warning: Tree is empty! please use another.")
        node = self.root
        while node and node.value != value:
            node = node.left if value < node.value else node.right
        return node

    def get_max(self) -> Node | None:
        """获取最大值节点"""
        node = self.root
        while node and node.right:
            node = node.right
        return node

    def get_min(self) -> Node | None:
        """获取最小值节点"""
        node = self.root
        while node and node.left:
            node = node.left
        return node

    def remove(self, value: int) -> None:
        """删除节点"""
        node = self.search(value)
        if node is None:
            raise ValueError(f"Value {value} not found")
        if node.left is None and node.right is None:  # 无子节点
            self._reassign_nodes(node, None)
        elif node.left is None:  # 只有右子节点
            self._reassign_nodes(node, node.right)
        elif node.right is None:  # 只有左子节点
            self._reassign_nodes(node, node.left)
        else:  # 有两个子节点
            predecessor = self.get_max(node.left)
            node.value = predecessor.value
            self.remove(predecessor.value)

    def _reassign_nodes(self, node: Node, new_children: Node | None) -> None:
        """重新分配节点的子节点"""
        if new_children:
            new_children.parent = node.parent
        if node.parent:
            if node.is_right:
                node.parent.right = new_children
            else:
                node.parent.left = new_children
        else:
            self.root = new_children

    def find_kth_smallest(self, k: int) -> int:
        """返回二叉搜索树中第 k 小的元素"""
        arr = []
        self._inorder(arr, self.root)
        return arr[k - 1]

    def _inorder(self, arr: list, node: Node | None) -> None:
        """中序遍历"""
        if node:
            self._inorder(arr, node.left)
            arr.append(node.value)
            self._inorder(arr, node.right)


if __name__ == "__main__":
    import doctest

    # 测试代码
    # 生活中的应用：组织学生成绩的树状结构
    scores = [85, 70, 90, 60, 80, 75, 95]
    bst = BinarySearchTree()
    for score in scores:
        bst.insert(score)

    print("学生成绩的最大值:", bst.get_max().value)  # 输出最大成绩
    print("学生成绩的最小值:", bst.get_min().value)  # 输出最小成绩
    print("第 3 小的成绩:", bst.find_kth_smallest(3))  # 输出第 3 小的成绩

    # 运行文档测试
    doctest.testmod(verbose=True)
