from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass

@dataclass
class Node:
    data: int
    left: Node | None = None
    right: Node | None = None

    def __iter__(self) -> Iterator[int]:
        # 先遍历左子树，再访问当前节点，最后遍历右子树
        if self.left:
            yield from self.left
        yield self.data
        if self.right:
            yield from self.right

    def is_full(self) -> bool:
        # 判断树是否为满二叉树
        if not self.left and not self.right:
            return True  # 叶子节点
        if self.left and self.right:
            return self.left.is_full() and self.right.is_full()  # 左右子树都存在
        return False  # 不满

@dataclass
class BinaryTree:
    root: Node

    def __iter__(self) -> Iterator[int]:
        return iter(self.root)

    def depth(self) -> int:
        # 计算树的深度
        return self._depth(self.root)

    def _depth(self, node: Node | None) -> int:
        if not node:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def is_full(self) -> bool:
        # 判断树是否为满二叉树
        return self.root.is_full()

    @classmethod
    def small_tree(cls) -> BinaryTree:
        """返回一个小的二叉树，包含3个节点"""
        return BinaryTree(Node(2, Node(1), Node(3)))

    @classmethod
    def medium_tree(cls) -> BinaryTree:
        """返回一个中等的二叉树，包含7个节点"""
        root = Node(4)
        root.left = Node(2, Node(1), Node(3))
        root.right = Node(5, None, Node(6, None, Node(7)))
        return BinaryTree(root)

if __name__ == "__main__":
    import doctest

    # 测试代码
    # 生活中的应用：组织家庭成员的树状结构
    # 例如：家庭树
    family_tree = BinaryTree.medium_tree()
    print("家庭树的深度:", family_tree.depth())  # 输出树的深度
    print("家庭树是否为满二叉树:", family_tree.is_full())  # 检查树是否为满二叉树

    # 运行文档测试
    doctest.testmod()
