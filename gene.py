class ConnectionGene:

    def __init__(self, input_node, output_node, innovation_number=None, enabled=True, weight=None, keep_constant_weight=False):
        self.innovation_number = innovation_number
        self.input_node = input_node
        self._output_node = output_node
        self.enabled = enabled
        self.weight = weight
        # This attribute is used in ghost nodes
        self.keep_constant_weight = keep_constant_weight

    @property
    def output_node(self):
        return self._output_node

    @output_node.setter
    def output_node(self, value):
        # You can't have a node which loops back to itself
        assert (value != self.input_node)
        self._output_node = value

    def __str__(self):
        return 'Input: {}, Output: {}'.format(self.input_node, self._output_node)

    def __repr__(self):
        return '{}-->{}'.format(self.input_node, self._output_node)


class NodeGene:

    def __init__(self, node_type, node_id, bias=None):
        # Specifies the type of node
        self._node_type = node_type
        # This is to keep track of which node is which
        self.node_id = node_id
        self.bias = bias

    @property
    def node_type(self):
        return self._node_type

    @node_type.setter
    def node_type(self, value):
        # There are only 3 possible types for a node gene
        assert (value in {'source', 'hidden', 'output'})
        self._node_type = value

    def __str__(self):
        return 'This is node number {} which is a {} node with a bias of {}'.format(self.node_id, self._node_type,
                                                                                    self.bias)

    def __repr__(self):
        return '{}:{}'.format(self.node_id, self._node_type)

    # def __add__(self, other):
    #     return self._node_number + other
