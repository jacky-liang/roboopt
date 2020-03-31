import torch
from torch_utils import zeros, clip


class Adam:

    def __init__(self, params, lo=None, hi=None, alpha=0.1, beta1=0.9, beta2=0.999, eps=1.e-8, wd=0):
        self._params = params
        self._lo = lo
        self._hi = hi
        self._grads = []

        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._wd = wd

        self._mt = zeros(len(self._params))
        self._vt = zeros(len(self._params))
        self._t = 0

    def _get_grad_update(self):
        grad = torch.mean(torch.stack(self._grads), dim=0)
        self._t += 1

        self._mt = self._beta1 * self._mt + (1 - self._beta1) * grad
        self._vt = self._beta2 * self._vt + (1 - self._beta2) * torch.pow(grad, 2)

        mt_hat = self._mt / (1 - (self._beta1**self._t))		
        vt_hat = self._vt / (1 - (self._beta2**self._t))

        return self._alpha * mt_hat / (torch.sqrt(vt_hat) + self._eps) + self._wd * self._params

    def step(self):
        update = self._get_grad_update()
        self._grads = []
        self._params = clip(self._params - update, lo=self._lo, hi=self._hi)

    def collect_grad(self, grad):
        self._grads.append(grad)

    @property
    def params(self):
        return self._params