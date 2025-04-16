import numpy as np
from typing import Optional
from src.tango.path_finding.graphs import PriorityQueue, CostMapGrid


class Search:
    def __init__(self, width: int, height: int, cost_map: Optional[np.ndarray] = None):
        self.frontier = PriorityQueue()
        self.graph = CostMapGrid(width=width, height=height, cost_map=cost_map)
        self.came_from = dict()
        self.cost_so_far = dict()
        self.dist_scaler = np.sqrt(width ** 2 + height ** 2)

    def get_path(self, start: tuple, goal: tuple) -> np.ndarray:
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = self.came_from[current]
        path.append(start)
        path.reverse()
        return np.array(path)

    def heuristic(self, current: tuple, goal: tuple) -> float:
        x1, y1 = current
        x2, y2 = goal
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2).item()  # prioritize movement that is closer to goal


class Dijkstra(Search):

    def search(self, start: tuple, goal: tuple):
        self.frontier.put(start, 0)
        self.came_from[start] = None
        self.cost_so_far[start] = 0
        while not self.frontier.empty():
            current = self.frontier.get()
            if current == goal:
                break
            for next in self.graph.neighbours(current):
                new_cost = self.cost_so_far[current] + self.graph.cost(current, next, start, goal)
                if next not in self.came_from or new_cost < self.cost_so_far[next]:
                    self.cost_so_far[next] = new_cost
                    priority = new_cost
                    self.frontier.put(next, priority)
                    self.came_from[next] = current


class AStar(Search):

    def search(self, start: tuple, goal: tuple):
        self.frontier.put(start, 0)
        self.came_from[start] = None
        self.cost_so_far[start] = 0

        while not self.frontier.empty():
            current: tuple = self.frontier.get()
            if current == goal:
                break
            for next in self.graph.neighbours(current):
                new_cost = self.cost_so_far[current] + self.graph.cost(current, next, start, goal)

                if next not in self.came_from or new_cost < self.cost_so_far[next]:
                    self.cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(start, current)
                    self.frontier.put(next, priority)
                    self.came_from[next] = current
