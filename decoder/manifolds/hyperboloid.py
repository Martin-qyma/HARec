"""Hyperboloid manifold. Copy from https://github.com/HazyResearch/hgcn """
import torch
from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature.
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            # add one dimension
            res = res.view(res.shape + (1,))
        return res

    def l_inner(self, x, y, keep_dim=False):
        # input shape [node, features]
        d = x.size(-1) - 1
        xy = x * y
        xy = torch.cat((-xy.narrow(1, 0, 1), xy.narrow(1, 1, d)), dim=1)
        return torch.sum(xy, dim=1, keepdim=keep_dim)

    def lorentzian_distance(self, x, y, c):
        # the squared Lorentzian distance
        xy_inner = self.l_inner(x, y)
        return -2 * (c + xy_inner)

    def sqdist_(self, p1, p2, c):
        dist = self.lorentzian_distance(p1, p2, c)
        dist = torch.clamp(dist, min = self.eps[p1.dtype], max=50)
        return dist

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)
    
    def dist_matrix(self, X, Y, c=torch.tensor([1.]).cuda()):
        K = 1. / c
        prod = self.minkowski_dot(X.unsqueeze(1), X.unsqueeze(0))
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[X.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def egrad2rgrad(self, x, grad, k, dim=-1):
        grad.narrow(-1, 0, 1).mul_(-1)
        grad = grad.addcmul(self.inner(x, grad, dim=dim, keepdim=True), x / k)
        return grad

    def inner(self, u, v, keepdim: bool = False, dim: int = -1):
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
                dim, 1, d
            ).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
                dim=dim, keepdim=True
            )
    
    def sphere_expmap0(self, u, k, dim=-1, min=1e-12):
        # assert torch.all(k) < 0
        x = u.narrow(-1, 1, u.size(-1) - 1)
        sqrtK = torch.sqrt(torch.abs(k))
        x_norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min)
        theta = x_norm / sqrtK

        l_v = sqrtK * torch.cos(theta)
        r_v = sqrtK * torch.sin(theta) * x / x_norm
        v = torch.cat((l_v, r_v), dim)
        return v

    def sphere_logmap0(self, x, k, dim=-1, min=1e-12):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        sqrtK = torch.sqrt(torch.abs(k))
        y_norm = torch.norm(y, p=2, dim=dim, keepdim=True).clamp(min)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + 1e-7)
        res[..., 1:] = sqrtK * torch.arccos(theta) * y / y_norm
        return res
        
    def oxy_angle(self, x, y, curv=1.0, eps=1e-8):
        """
        Given two vectors `x` and `y` on the hyperboloid, compute the exterior
        angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
        of the hyperboloid.

        This expression is derived using the Hyperbolic law of cosines.

        Args:
            x: Tensor of shape `(B, D)` giving a batch of space components of
                vectors on the hyperboloid.
            y: Tensor of same shape as `x` giving another batch of vectors.
            curv: Positive scalar denoting negative hyperboloid curvature.

        Returns:
            Tensor of shape `(B, )` giving the required angle. Values of this
            tensor lie in `(0, pi)`.
        """

        # Calculate time components of inputs (multiplied with `sqrt(curv)`):
        x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
        y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

        # Calculate lorentzian inner product multiplied with curvature. We do not use
        # the `pairwise_inner` implementation to save some operations (since we only
        # need the diagonal elements).
        c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

        # Make the numerator and denominator for input to arc-cosh, shape: (B, )
        acos_numer = y_time + c_xyl * x_time
        acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

        acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
        _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

        return _angle

    def half_aperture(self, x, curv=1.0, min_radius=0.1, eps=1e-8):
        """
        Compute the half aperture angle of the entailment cone formed by vectors on
        the hyperboloid. The given vector would meet the apex of this cone, and the
        cone itself extends outwards to infinity.

        Args:
            x: Tensor of shape `(B, D)` giving a batch of space components of
                vectors on the hyperboloid.
            curv: Positive scalar denoting negative hyperboloid curvature.
            min_radius: Radius of a small neighborhood around vertex of the hyperboloid
                where cone aperture is left undefined. Input vectors lying inside this
                neighborhood (having smaller norm) will be projected on the boundary.
            eps: Small float number to avoid numerical instability.

        Returns:
            Tensor of shape `(B, )` giving the half-aperture of entailment cones
            formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
        """

        # Ensure numerical stability in arc-sin by clamping input.
        asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
        _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

        return _half_aperture


