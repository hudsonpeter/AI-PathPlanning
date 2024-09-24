"""Search Algorithms"""
from collections import deque
import itertools
import math
from queue import PriorityQueue
import time

START = "start"
GOAL = "goal"
ALGO_BFS = "BFS"
ALGO_UCS = "UCS"
ALGO_A_STAR = "A*"

counter = itertools.count()


class Node:
    def __init__(self, name, path, parent=None, momentum=0, path_cost=0):
        self.name = name  # node name
        self.path = path  # list to track the paths traversed
        self.parent = parent  # parent node
        self.momentum = momentum  # momentum gained after reaching the curretn location
        self.path_cost = path_cost  # total path cost to reaching the


def parse_input(file_path):
    """parse_input: parses input file"""
    with open(file_path, "r") as file:
        # first line - algorithm
        algo = file.readline().strip()

        # second line - rover uphill energy limit
        uphill_energy = int(file.readline().strip())

        # third line - number of safe locations (nodes)
        number_of_safe_locations = int(file.readline().strip())

        # next num_of_safe_locations lines - Safe locations
        safe_locations = {}
        for _ in range(number_of_safe_locations):
            line = file.readline().strip().split()
            location, x, y, z = line[0], int(line[1]), int(line[2]), int(line[3])
            safe_locations[location] = (x, y, z)

        # next line - Number of safe path segments (edges)
        number_of_safe_paths = int(file.readline().strip())

        # adjacency list for safe paths
        safe_paths = {}
        for _ in range(number_of_safe_paths):
            line = file.readline().strip().split()
            node1, node2 = line[0], line[1]

            if node1 not in safe_paths:
                safe_paths[node1] = []
            if node2 not in safe_paths:
                safe_paths[node2] = []

            safe_paths[node1].append(node2)
            safe_paths[node2].append(node1)

    # print("File operations completed")
    return algo, uphill_energy, safe_locations, safe_paths


def bfs(safe_locations, adjacency_list, rover_energy):
    # print("Start of BFS")
    start_node = Node(START, [START])
    frontier = deque([start_node])
    visited = {}

    while frontier:
        current_node = frontier.popleft()
        current_node_name = current_node.name
        parent_z = (
            0
            if current_node.parent not in visited
            else safe_locations[current_node.parent][2]
        )
        current_z = safe_locations[current_node_name][2]

        momentum = max(0, parent_z - current_z)

        if (
            current_node_name in visited
            and visited[current_node_name].momentum >= momentum
        ):
            continue

        if current_node_name == GOAL:
            return current_node.path

        current_node.momentum = momentum
        visited[current_node_name] = current_node

        neighbours = [
            Node(neighbour, current_node.path[:], current_node.name, 0)
            for neighbour in adjacency_list.get(current_node_name)
        ]

        for neighbour in neighbours:
            neighbour_z = safe_locations[neighbour.name][2]
            is_energy_sufficient = neighbour_z - current_z <= (rover_energy + momentum)
            if is_energy_sufficient:
                neighbour.path.append(neighbour.name)
                frontier.append(neighbour)

    # print("BFS completed without paths")
    return []


def euclidean_distance(location1, location2, algo):
    """Calculates Eulidean distance for the locations passed as params
    algo decides if the euclidean distance to be calculated to be 2D or 3D"""
    squared_diff = []
    if algo == ALGO_UCS:
        squared_diff = [
            (coord1 - coord2) ** 2
            for coord1, coord2 in zip(location1[:2], location2[:2])
        ]
    else:
        squared_diff = [
            (coord1 - coord2) ** 2 for coord1, coord2 in zip(location1, location2)
        ]
    distance = math.sqrt(sum(squared_diff))
    # print(distance)
    return distance


def ucs(safe_locations, adjacency_list, rover_energy):
    # print("Start of UCS")
    start_node = (
        0,
        next(counter),
        Node(START, [START], None, 0),
    )  # counter used as a tie breaker for same cost
    frontier = PriorityQueue()
    frontier.put(start_node)

    reached = {}

    while frontier:
        current_node = frontier.get()[2]
        current_node_name = current_node.name
        parent_z = (
            0
            if current_node.parent is None or current_node.parent not in reached
            else safe_locations[current_node.parent][2]
        )

        current_z = safe_locations[current_node_name][2]

        momentum = max(0, parent_z - current_z)

        if (
            current_node_name in reached
            and reached[current_node_name].momentum >= momentum
        ):
            continue

        current_node.momentum = momentum
        reached[current_node_name] = current_node

        if current_node_name == GOAL:
            return (current_node.path, current_node.path_cost)

        neighbours = [
            Node(neighbour, current_node.path[:], current_node_name)
            for neighbour in adjacency_list.get(current_node_name)
        ]

        for neighbour in neighbours:
            neighbour_z = safe_locations[neighbour.name][2]
            is_energy_sufficient = neighbour_z - current_z <= (rover_energy + momentum)
            neighbour.path_cost = (
                euclidean_distance(
                    safe_locations[neighbour.name],
                    safe_locations[current_node_name],
                    ALGO_UCS,
                )
                + current_node.path_cost
            )

            if is_energy_sufficient:
                neighbour.path.append(neighbour.name)
                neighbour_node = (neighbour.path_cost, next(counter), neighbour)
                frontier.put(neighbour_node)

    # print("UCS completed without paths")
    return ([], 0)


def heuristic(current_node, goal_node):
    """Heuristic for A* Algorithm"""
    return euclidean_distance(
        current_node,
        goal_node,
        ALGO_A_STAR,
    )


def a_star(safe_locations, adjacency_list, rover_energy):
    # print("Start of A*")
    start_node = (
        0,
        next(counter),
        Node(START, [START], None, 0),
    )  # counter used as a tie breaker for same cost
    frontier = PriorityQueue()
    frontier.put(start_node)

    reached = {}

    while frontier:
        current_node = frontier.get()[2]
        current_node_name = current_node.name
        parent_z = (
            0
            if current_node.parent is None or current_node.parent not in reached
            else safe_locations[current_node.parent][2]
        )

        current_z = safe_locations[current_node_name][2]

        momentum = max(0, parent_z - current_z)

        if (
            current_node_name in reached
            and reached[current_node_name].momentum >= momentum
        ):
            continue

        current_node.momentum = momentum
        reached[current_node_name] = current_node

        if current_node_name == GOAL:
            return (current_node.path, current_node.path_cost)

        neighbours = [
            Node(neighbour, current_node.path[:], current_node_name)
            for neighbour in adjacency_list.get(current_node_name)
        ]

        for neighbour in neighbours:
            neighbour_z = safe_locations[neighbour.name][2]
            is_energy_sufficient = neighbour_z - current_z <= (rover_energy + momentum)
            neighbour.path_cost = (
                euclidean_distance(
                    safe_locations[neighbour.name],
                    safe_locations[current_node_name],
                    ALGO_A_STAR,
                )
                + current_node.path_cost
            )

            if is_energy_sufficient:
                neighbour.path.append(neighbour.name)
                neighbour_node = (
                    neighbour.path_cost
                    + heuristic(
                        safe_locations[neighbour.name],
                        safe_locations[GOAL],
                    ),
                    next(counter),
                    neighbour,
                )
                frontier.put(neighbour_node)

    # print("A* completed without paths")
    return ([], 0)


# start = time.time()

algorithm, uphill_energy_limit, locations, paths = parse_input("input.txt")

computed_path = []
path_cost = 0

# match algorithm:
#     case "BFS":
#         computed_path = bfs(locations, paths, uphill_energy_limit)
#     case "UCS":
#         computed_path, path_cost = ucs(locations, paths, uphill_energy_limit)
#     case "A*":
#         computed_path, path_cost = a_star(locations, paths, uphill_energy_limit)
#     case _:
#         print("FAIL")

if algorithm == "BFS":
    computed_path = bfs(locations, paths, uphill_energy_limit)
elif algorithm == "UCS":
    computed_path, path_cost = ucs(locations, paths, uphill_energy_limit)
elif algorithm == "A*":
    computed_path, path_cost = a_star(locations, paths, uphill_energy_limit)
else:
    print("FAIL")

with open("output.txt", "w") as output_file:
    if computed_path is None or len(computed_path) == 0:
        output_file.write("FAIL")
    else:
        output_file.write((" ").join(computed_path))
    output_file.close()

# print("path: ", (" ").join(computed_path))
# print("pathlen: ", len(computed_path) - 1)
# print("path_cost: ", round(path_cost, 2))
# end = time.time()
# print("The time of execution of above program is :", (end - start), "s")
