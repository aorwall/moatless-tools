from pydantic import BaseModel, Field


class PathTree(BaseModel):
    show: bool = Field(default=False, description="Show the block.")
    tree: dict[str, "PathTree"] = Field(default_factory=dict)

    def merge(self, other: "PathTree"):
        if other.show:
            self.show = True

        for key, value in other.tree.items():
            if key not in self.tree:
                self.tree[key] = PathTree()
            self.tree[key].merge(value)

    def add_to_tree(self, path: list[str]):
        if path is None:
            return

        if len(path) == 0:
            self.show = True
            return

        if len(path) == 1:
            if path[0] not in self.tree:
                self.tree[path[0]] = PathTree(show=True)
            else:
                self.tree[path[0]].show = True

            return

        if path[0] not in self.tree:
            self.tree[path[0]] = PathTree(show=False)

        self.tree[path[0]].add_to_tree(path[1:])
