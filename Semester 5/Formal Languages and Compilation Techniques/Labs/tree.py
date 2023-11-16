class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.index = 0

    def insert(self, root, data):
        if root is None:
            root = Node(data)
        else:
            if root.data < data:
                root.right = self.insert(root.right, data)
            else:
                root.left = self.insert(root.left, data)
        return root


class Tree:
    def __init__(self):
        self.root = None
        self.TS_code = None

    def insert(self, data):
        if self.root is None:
            self.root = Node(data)
        else:
            self.root.insert(self.root, data)

    def print_tree(self):
        self.TS_code = 0
        self.print_tree_rec(self.root)

    def print_tree_rec(self, root):
        if root is None:
            return
        self.print_tree_rec(root.left)
        print(root.data)
        root.index = self.TS_code
        self.TS_code += 1
        self.print_tree_rec(root.right)

    def search(self, value):
        return self.search_rec(self.root, value)

    def search_rec(self, root, value):
        if root is None:
            return False
        if root.data == value:
            return True
        if root.data < value:
            return self.search_rec(root.right, value)
        return self.search_rec(root.left, value)

    def get_index(self, value):
        return self.get_index_rec(self.root, value)

    def get_index_rec(self, root, value):
        if root is None:
            return -1
        if root.data == value:
            return root.index
        if root.data < value:
            return self.get_index_rec(root.right, value)
        return self.get_index_rec(root.left, value)
