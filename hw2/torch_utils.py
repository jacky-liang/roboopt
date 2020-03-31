import os
import random
import numpy as np
import torch

from torch.autograd import Function
import scipy.linalg


# From https://github.com/steveli/pytorch-sqrtm
class _MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, m):
        m = get_numpy(m)
        sqrtm = from_numpy(scipy.linalg.sqrtm(m).real)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = get_numpy(sqrtm)
            gm = get_numpy(grad_output)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = from_numpy(grad_sqrtm)
        return grad_input


sqrtm = _MatrixSquareRoot.apply


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


def set_seed(seed, do_torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)

    if do_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_apply(f, X):
    return torch.stack([f(x) for x in X])


def clip(x, lo=None, hi=None):
    if lo is not None:
        x = torch.relu(x - lo) + lo
    if hi is not None:
        x = -torch.relu(-x + hi) + hi
    return x


"""
GPU wrappers from 
https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
"""

_use_gpu = False
device = None

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)

def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs).float().to(device)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()

def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)

def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)

def eye(*sizes, **kwargs):
    return torch.eye(*sizes, **kwargs).to(device)

def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)

def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs).to(device)

def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)