import heapq
import torch
import numpy as np
import networkx as nx
from typing import Optional, Iterator


# guided by: https://www.redblobgames.com/pathfinding/a-star/implementation.html
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class CostMapGrid:

    def __init__(self, width: int, height: int, cost_map: Optional[np.ndarray] = None):
        self.width = width
        self.height = height
        self.weights = dict()
        self.cost_map = cost_map
        self.max_field_cost = self.cost_map.max()
        self.max_distance = self.width + self.height
        self.grad_x, self.grad_y = torch.gradient(torch.tensor(self.cost_map))

    def in_bounds(self, coordinates: tuple) -> bool:
        (x, y) = coordinates
        return 0 <= x < self.width - 1 and 0 <= y < self.height - 1

    def neighbours(self, coordinates: tuple) -> Iterator[tuple]:
        (x, y) = coordinates
        neighbours = [
            (x, y + 1),
            (x + 1, y),
            (x, y - 1),
            (x - 1, y),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
        ]
        if (x + y) % 2 == 0:
            neighbours.reverse()
        results = filter(self.in_bounds, neighbours)
        return results

    def cost(self, current_node, next_node, start_node, goal_node) -> float:
        x0, y0 = start_node
        x1, y1 = next_node
        x2, y2 = current_node
        x3, y3 = goal_node
        cost_field = self.cost_map[y1, x1] + np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # diagonal costs slightly more
        return self.weights.get(next_node, cost_field)


class CostMapGraphNX:

    def __init__(self, width: int, height: int, cost_map: np.ndarray):
        self.width = width
        self.height = height
        self.cost_map = cost_map - cost_map.min()
        self.graph = self.build_graph()

    # @jit(forceobj=True, looplift=True)
    def build_graph(self):
        xs, ys = np.meshgrid(range(self.width), range(self.height))
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        graph = nx.Graph()
        # todo: this is slow make faster @ 0.056ms right now
        for x, y in zip(xs, ys):
            for neighbours in self.neighbours((x, y)):
                x_neighbour, y_neighbour = neighbours
                distance = np.sqrt((x - x_neighbour) ** 2 + (y - y_neighbour) ** 2)
                cost = self.cost_map[y_neighbour, x_neighbour] + distance
                graph.add_edges_from([(f'{x},{y}', f'{x_neighbour},{y_neighbour}')], weight=cost)

        return graph

    def in_bounds(self, coordinates: tuple) -> bool:
        (x, y) = coordinates
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbours(self, coordinates: tuple) -> Iterator[tuple]:
        (x, y) = coordinates

        neighbours = [
            (x, y + 1),
            (x + 1, y),
            (x, y - 1),
            (x - 1, y),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
        ]
        results = filter(self.in_bounds, neighbours)
        return results

    def get_path(self, start: tuple, goal: tuple) -> np.ndarray:
        shortest_path = [
            coord.split(',') for coord in nx.dijkstra_path(
                self.graph, f'{start[0]},{start[1]}', f'{goal[0]},{goal[1]}',
                weight='weight'
            )]

        shortest_path = np.array(shortest_path, dtype=float)
        return shortest_path
