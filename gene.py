class ConnectionGene:

    def __init__(self, input_node, output_node, innovation_number=None, enabled=True, weight=None):
        self._innovation_number = innovation_number
        self._input_node = input_node
        self._output_node = output_node
        self._enabled = enabled
        self._weight = weight

    @property
    def weight(self):
        return self._weight

    @property
    def innovation_number(self):
        return self._innovation_number

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._output_node

    @property
    def enabled(self):
        return self._enabled


class NodeGene:

    def __init__(self, node_type, node_id, bias=None):

        # There are only 3 possible types for a node gene
        assert (node_type in {'source', 'hidden', 'output'})

        # Specifies the type of node
        self._node_type = node_type
        # This is to keep track of which node is which
        self._node_id = node_id
        self._bias = bias

    @property
    def bias(self):
        return self._bias

    @property
    def node_type(self):
        return self._node_type

    @property
    def node_id(self):
        return self._node_id

    def __str__(self):
        return 'This is node number {} which is a {} node with a bias of {}'.format(self._node_id, self._node_type, self._bias)

    # def __add__(self, other):
    #     return self._node_number + other
