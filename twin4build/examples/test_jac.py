import torch

def rev_jacobian(fxn, x, n_outputs, retain_graph):
    """
    the basic idea is to create N copies of the input
    and then ask for each of the N dimensions of the
    output... this allows us to compute J with pytorch's
    jacobian-vector engine
    """

    # expand the input, one copy per output dimension
    n_outputs = int(n_outputs)
    repear_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repear_arg)
    xr.requires_grad_(True)

    # both y and I are shape (n_outputs, n_outputs)
    #  checking y shape lets us report something meaningful
    y = fxn(xr).view(n_outputs, -1)

    print("x.size() ", x.size())
    print("n_outputs ", n_outputs)
    print("repear_arg ", repear_arg)
    print("x.shape ", x.shape)
    print("xr.shape ", xr.shape)
    print("y.shape ", y.shape)
    print("y.size() ", y.size())

    if y.size(1) != n_outputs: 
        raise ValueError('Function `fxn` does not give output '
                         'compatible with `n_outputs`=%d, size '
                         'of fxn(x) : %s' 
                         '' % (n_outputs, y.size(1)))
    I = torch.eye(n_outputs, device=xr.device)

    J = torch.autograd.grad(y, xr,
                      grad_outputs=I,
                      retain_graph=retain_graph,
                      create_graph=True,  # for higher order derivatives
                      )

    return J[0]

def fxn(x):
    return x**2 + x[0]*x[1]


def test_jac():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    n_outputs = 3
    retain_graph = True
    J = rev_jacobian(fxn, x, n_outputs, retain_graph)
    print(J)

if __name__ == "__main__":
    test_jac()