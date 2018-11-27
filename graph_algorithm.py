import copy
class Graph:
    """
    Class used to find the number of paths between two nodes in a grsph
    """

    def __init__(self):
        # Keeps track of the vertex's in the graph
        self.vertex_list = []
        # Keeps track of the connections for each node
        self.connections = {}

    def count_path_utils(self, current_node, destination, visited, path_count, path, overall_paths):
        """
        Checks if we're at the destination, adds one if we are, if not check all the neighbours of the current node
        :param current_node: The node we're currently add
        :param destination: The end of the expected path
        :param visited: A dict which keeps which nodes have been visited or not
        :param path_count: Keeps the number of paths from the designated start and end node
        :return:
        """
        # We've visited the current node since we're at it
        visited[current_node] = True
        path.append(current_node)

        # If the current node is the destination then we can increas the path_count number
        if current_node == destination:
            path_count.append(1)
            overall_paths.append(copy.deepcopy(path))
        else:
            # Go through all the neighbours looking for the destination
            for neighbour in self.connections[current_node]:
                # If we haven't visited the neighbour, look through the neighbour for the destination
                if not visited[neighbour]:
                    # Call the function recursively
                    self.count_path_utils(neighbour, destination, visited, path_count, path, overall_paths)
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        # Once we've checked all the neighbour's we can set the visited to false again
        visited[current_node] = False

    def add_edge(self, start_node, end_node):
        connection_dict = self.connections.get(start_node)
        if connection_dict:
            connection_dict.append(end_node)
        else:
            self.connections[start_node] = [end_node]
            self.vertex_list.append(start_node)
            self.vertex_list.append(end_node)

    def count_paths(self, start_node, end_node, return_paths=False):
        """
        Count paths from start_node to end_node
        :param start_node: Where the path starts
        :param end_node: Where the end of the path is
        :return:
        """

        # Keep's track of a node has been visited or not
        visited = {node: False for node in self.vertex_list}
        paths = []
        overall_paths = []
        path_count = []
        self.count_path_utils(start_node, end_node, visited, path_count, paths, overall_paths)
        if return_paths:
            return sum(path_count), overall_paths
        else:
            return sum(path_count)


def main():
    g = Graph()
    g.add_edge(2, 3)
    g.add_edge(3, 5)

    print(g.count_paths(2, 5, True))


if __name__ == "__main__":
    main()
