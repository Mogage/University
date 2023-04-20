import networkx as nx


class DynamicGraph(nx.Graph):
    def __init__(self, time=0, **kwargs):
        super().__init__(**kwargs)
        self.time = time
        self.edge_times = {}

    def add_edge(self, u, v, time=0, **kwargs):
        self.edge_times[(u, v)] = time
        super().add_edge(u, v, **kwargs)

    def remove_edge(self, u, v):
        if (u, v) in self.edge_times:
            del self.edge_times[(u, v)]
        super().remove_edge(u, v)

    def update_time(self, time):
        # Remove edges that are older than the new time
        for (u, v), edge_time in list(self.edge_times.items()):
            if edge_time < time:
                self.remove_edge(u, v)
        self.time = time
