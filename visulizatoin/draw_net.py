from graphviz import Digraph
from torch.autograd import Variable
import torch


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    # visulize the netwok or drwa the network or show the network
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


if __name__ == '__main__':

    # output the shape of of every layers' feature maps
    from models.layers import Conv, Residual, Hourglass

    # ##############################################################3
    # from torchsummary import summary
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    #
    # model = Hourglass2(2, 32, 1, Residual).to(device)
    # t = model._make_hour_glass()
    # for i in t.named_modules():
    #     print(i)
    # summary(model, (32, 128, 128))
    # ##############################################################3

    # ##############################################################3

    #
    # # plot the models
    # model = Hourglass2(4, 32, 1, Conv)
    # x = Variable(torch.randn(1, 32, 128, 128))  # x的shape为(batch，channels，height，width)
    # y = model(x)
    # g = make_dot(y)
    # g.view()
    # ##############################################################3

    import torch.onnx
    net = Hourglass(4, 256,  128,  resBlock=Conv)
    dummy_input = torch.randn(1, 256, 128, 128)
    torch.onnx.export(net, dummy_input, "hourglass.onnx")






