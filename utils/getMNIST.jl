using PyCall

function getMNISTpy()
    py"""
    # coding: utf-8
    import sys, os
    sys.path.append(os.pardir) 
    import pickle
    from datasets.mnist import load_mnist

    def get_data():
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_train, t_train, x_test, t_test

    x_train, t_train, x_test, t_test = get_data()
    """

    return py"x_train", py"t_train", py"x_test", py"t_test"
end