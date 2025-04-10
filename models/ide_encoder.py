import numpy as np
import torch
import torch.nn as nn

def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    # return np.prod(a - np.arange(k)) / np.math.factorial(k)
    return np.prod(a - np.arange(k)) / np.math.factorial(k)

def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

    Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return (
        (-1) ** m
        * 2**l
        * np.math.factorial(l)
        / np.math.factorial(k)
        / np.math.factorial(l - k - m)
        * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l)
    )

def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    # return (np.sqrt(
    #     (2.0 * l + 1.0) * np.math.factorial(l - m) /
    #     (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))
    return np.sqrt(
        (2.0 * l + 1.0)
        * np.math.factorial(l - m)
        / (4.0 * np.pi * np.math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    ml_array = np.array(ml_list).T
    return ml_array

class IntegratedDirEncoder(nn.Module):
    """Module for integrated directional encoding (IDE).
        from Equations 6-8 of arxiv.org/abs/2112.03907.
    """

    def __init__(self, input_dim=3, deg_view=4):
        """Initialize integrated directional encoding (IDE) module.

        Args:
            deg_view: number of spherical harmonics degrees to use.
        
        Raises:
            ValueError: if deg_view is larger than 5.

        """
        super().__init__()
        self.deg_view = deg_view

        if deg_view > 5:
            raise ValueError("Only deg_view of at most 5 is numerically stable.")

        ml_array = get_ml_array(deg_view)
        l_max = 2 ** (deg_view - 1)

        # Create a matrix corresponding to ml_array holding all coefficients, which,
        # when multiplied (from the right) by the z coordinate Vandermonde matrix,
        # results in the z component of the encoding.
        mat = np.zeros((l_max + 1, ml_array.shape[1]))
        for i, (m, l) in enumerate(ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = sph_harm_coeff(l, m, k)

        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
    
        self.register_buffer("mat", torch.Tensor(mat), False)
        self.register_buffer("ml_array", torch.Tensor(ml_array), False)
        self.register_buffer("pow_level", torch.arange(l_max + 1), False)
        self.register_buffer("sigma", torch.Tensor(sigma), False)

        self.n_input_dims = input_dim
        self.n_output_dims = (2**deg_view - 1 + deg_view) * 2

    def forward(self, xyz, roughness=0, **kwargs):
        """Compute integrated directional encoding (IDE).

        Args:
            xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
            kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
                Mises-Fisher distribution.

        Returns:
            An array with the resulting IDE.
        """
        kappa_inv = roughness
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]
        # avoid 0 + 0j exponentiation
        zero_xy = torch.logical_and(x == 0, y == 0)
        y = y + zero_xy

        vmz = z ** self.pow_level
        vmxy = (x + 1j * y) ** self.ml_array[0, :]

        sph_harms = vmxy * torch.matmul(vmz, self.mat)
        ide = sph_harms * torch.exp(-self.sigma * kappa_inv)

        # check whether Nan appears
        if torch.isnan(ide).any():
            print('Nan appears in IDE')
            import IPython; IPython.embed()
            raise ValueError('Nan appears in IDE')

        return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)       
    
    def forward_wo_j(self, xyz, roughness=0, **kwargs): # a non-complex version for web demo
        """Compute integrated directional encoding (IDE).

        Args:
            xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
            kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
                Mises-Fisher distribution.

        Returns:
            An array with the resulting IDE.
        """
        kappa_inv = roughness
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]
        # avoid 0 + 0j exponentiation
        zero_xy = torch.logical_and(x == 0, y == 0)
        y = y + zero_xy

        vmz = z ** self.pow_level
        # vmxy = (x + 1j * y) ** self.ml_array[0, :]
        # euler's formula: e^(i theta) = cos(theta) + i sin(theta)
        vmxy_r = torch.pow(x ** 2 + y ** 2, self.ml_array[0, :] / 2)
        vmxy_theta = torch.atan2(y, x) * self.ml_array[0, :]
        vmxy_x = vmxy_r * torch.cos(vmxy_theta) # real part
        vmxy_y = vmxy_r * torch.sin(vmxy_theta) # imaginary part
        
        z_component = torch.matmul(vmz, self.mat)
        sph_harms_x = vmxy_x * z_component
        sph_harms_y = vmxy_y * z_component

        exp_scale = torch.exp(-self.sigma * kappa_inv)
        ide_x = sph_harms_x * exp_scale
        ide_y = sph_harms_y * exp_scale

        return torch.cat([ide_x, ide_y], dim=-1)     

if __name__ == "__main__":
    # Example usage.
    # xyz = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    xyz = torch.tensor([[0.1, 0.2, 0.3], [0, 0, 0.6], [0.7, 0.8, 0.9]])
    
    # xyz = torch.rand(3, 3) * 2 - 1
    kappa_inv = torch.ones(3)[..., None]
    ide = IntegratedDirEncoder(4)
    dir_enc = ide(xyz, kappa_inv)
    dir_enc_wo_j = ide.forward_wo_j(xyz, kappa_inv)
    print(dir_enc.shape)
    print(dir_enc_wo_j.shape)
    print((dir_enc - dir_enc_wo_j).abs().max())
