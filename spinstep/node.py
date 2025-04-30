class Node:
    def __init__(self, name, orientation, children=None):
        self.name = name
        self.orientation = orientation  # [x, y, z, w]
        self.children = children if children else []
