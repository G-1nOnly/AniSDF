import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.utilities.rank_zero import rank_zero_info

import models
from models.base import BaseModel
from models.utils import scale_anything, get_activation, cleanup, chunk_batch
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network
from utils.misc import get_rank
from systems.utils import update_module_step
from nerfacc import ContractionType

import math


def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch')
        self.radius = self.config.radius
        self.contraction_type = None # assigned in system

    def forward_level(self, points):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax):
        def batch_func(x):
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv
    
        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices())
        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    @torch.no_grad()
    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius))
        vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        mesh_fine = self.isosurface_(vmin_, vmax_)
        return mesh_fine 

class MarchingCubeHelper_angelo(nn.Module):
    def __init__(self, sdf_func, bounds, resolution, block_res=512, method='mc'):
        super().__init__()
        self.sdf_func = sdf_func
        self.bounds = bounds
        self.resolution = resolution
        self.intv = 2.0 / self.resolution
        self.block_res = block_res
        self.points_range = (0, 1)
        self.method = method
        try:
            import cumcubes
        except:
            print("Cannot find cuda accelerated marching cube, downgraded to cpu version!")
            self.method = 'mc'
 
        if self.method == 'CuMCubes':
            self.mc_func = cumcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None
        self._create_lattice_grid()

    def _create_lattice_grid(self):
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = self.bounds
        self.x_grid = torch.arange(x_min, x_max, self.intv)
        self.y_grid = torch.arange(y_min, y_max, self.intv)
        self.z_grid = torch.arange(z_min, z_max, self.intv)
        res_x, res_y, res_z = len(self.x_grid), len(self.y_grid), len(self.z_grid)
        print("Extracting surface at resolution", res_x, res_y, res_z)
        self.num_blocks_x = int(np.ceil(res_x / self.block_res))
        self.num_blocks_y = int(np.ceil(res_y / self.block_res))
        self.num_blocks_z = int(np.ceil(res_z / self.block_res))
        
    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward_(self, level, threshold=0.):
        if self.method == 'CuMCubes':
            verts, faces = self.mc_func(-level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.cpu().numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        return verts, faces
    
    def forward(self, threshold=0.):
        mesh_blocks = []
        for idx in range(self.num_blocks_x * self.num_blocks_y * self.num_blocks_z):
            block_idx_x = idx // (self.num_blocks_y * self.num_blocks_z)
            block_idx_y = (idx // self.num_blocks_z) % self.num_blocks_y
            block_idx_z = idx % self.num_blocks_z
            xi = block_idx_x * self.block_res
            yi = block_idx_y * self.block_res
            zi = block_idx_z * self.block_res
            x, y, z = torch.meshgrid(self.x_grid[xi:xi+self.block_res+1],
                                    self.y_grid[yi:yi+self.block_res+1],
                                    self.z_grid[zi:zi+self.block_res+1], indexing="ij")
            xyz = torch.stack([x, y, z], dim=-1)
            sdf = self.sdf_func(xyz.cuda())
            verts, faces = self.forward_(sdf, threshold)
            if verts.shape[0] > 0:
                verts = verts * self.intv + xyz[0, 0, 0]
                mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
            else:
                mesh = trimesh.Trimesh()
            mesh_blocks.append(mesh)
        mesh = trimesh.util.concatenate(mesh_blocks)
        return {
            'v_pos': torch.from_numpy(np.array(mesh.vertices)),
            't_pos_idx': torch.from_numpy(np.array(mesh.faces))
        }
        
class BaseImplicitGeometry_angelo(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.radius = self.config.radius
        self.contraction_type = None # assigned in system
        self.sdf_func = lambda x: -self.forward_level(x)
        self.bounds = np.array([[-self.radius, self.radius], [-self.radius, self.radius], [-self.radius, self.radius]])
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'CuMCubes']
            self.helper = MarchingCubeHelper_angelo(self.sdf_func, self.bounds, int(self.config.isosurface.resolution), method=self.config.isosurface.method)

    def forward_level(self, points):
        raise NotImplementedError

    @torch.no_grad()
    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh = self.helper(threshold=0.001)
        return mesh



@models.register('volume-density')
class VolumeDensity(BaseImplicitGeometry):
    def setup(self):
        self.n_input_dims = self.config.get('n_input_dims', 3)
        self.n_output_dims = self.config.feature_dim
        self.encoding_with_network = get_encoding_with_network(self.n_input_dims, self.n_output_dims, self.config.xyz_encoding_config, self.config.mlp_network_config)

    def forward(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type)
        out = self.encoding_with_network(points.view(-1, self.n_input_dims)).view(*points.shape[:-1], self.n_output_dims).float()
        density, feature = out[...,0], out
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        if 'feature_activation' in self.config:
            feature = get_activation(self.config.feature_activation)(feature)
        return density, feature

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type)
        density = self.encoding_with_network(points.reshape(-1, self.n_input_dims)).reshape(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'density_activation' in self.config:
            density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        return -density      

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding_with_network, epoch, global_step)


@models.register('volume-sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
                
                out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                sdf, feature = out[...,0], out
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  

                        if with_laplace:
                            laplace = (points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]).sum(-1) / (eps ** 2)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == 'finite_difference', "Laplace computation is only supported with grad_type='finite_difference'"
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        if self.grad_type == 'finite_difference':
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(f"Update finite_difference_eps to {grid_size}")
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")


@models.register('volume-sdf-laplace')
class VolumeSDFL(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False, with_auxiliary_feature=False):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
                
                out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                sdf, feature = out[...,0], out
                feature = torch.concat([feature, (points * 2 - 1)], dim=-1)
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  
                
                if with_laplace or with_auxiliary_feature:
                    eps=self._finite_difference_eps
                    rand_directions=torch.randn_like(points)
                    rand_directions=F.normalize(rand_directions,dim=-1)

                    #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
                    normals=F.normalize(grad,dim=-1)
                    tangent=torch.cross(normals, rand_directions)
                    rand_directions=tangent #set the random moving direction to be the tangent direction now
                    
                    points_shifted=points.clone()+rand_directions*eps

                if with_auxiliary_feature:
                    points_shifted_ = scale_anything(points_shifted, (-self.radius, self.radius), (0, 1))
                    points_shifted_feature = self.network(self.encoding(points_shifted_.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                    if 'feature_activation' in self.config:
                        points_shifted_feature = get_activation(self.config.feature_activation)(points_shifted_feature)
                if with_laplace:                    
                    offsets = torch.as_tensor(
                    [
                        [eps, 0.0, 0.0],
                        [-eps, 0.0, 0.0],
                        [0.0, eps, 0.0],
                        [0.0, -eps, 0.0],
                        [0.0, 0.0, eps],
                        [0.0, 0.0, -eps],
                    ]
                    ).to(points_)
                    points_shifted_d_ = (points_shifted[...,None,:] + offsets).clamp(-self.radius, self.radius)
                    points_shifted_d = scale_anything(points_shifted_d_, (-self.radius, self.radius), (0, 1))
                    points_shifted_d_sdf = self.network(self.encoding(points_shifted_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                    sdf_gradients_shifted = 0.5 * (points_shifted_d_sdf[..., 0::2] - points_shifted_d_sdf[..., 1::2]) / eps  

                    normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

                    dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
                    #the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
                    angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and pi when is opposite

                    laplace=angle/math.pi #map to [0,1 range]

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            rv.append(laplace)
        if with_auxiliary_feature:
            rv.append(points_shifted_feature)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        if isinstance(self.finite_difference_eps, float):
            self._finite_difference_eps = self.finite_difference_eps
        elif self.finite_difference_eps == 'progressive':
            hg_conf = self.config.xyz_encoding_config
            assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
            current_level = min(
                hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                hg_conf.n_levels
            )
            grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
            grid_size = 2 * self.config.radius / grid_res
            if grid_size != self._finite_difference_eps:
                rank_zero_info(f"Update finite_difference_eps to {grid_size}")
            self._finite_difference_eps = grid_size
        else:
            raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")
        
@models.register('volume-sdf-laplace-fusion')
class VolumeSDFLF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        encoding_f = get_encoding(3, self.config.xyz_encoding_config_f)
        network_f = get_mlp(encoding_f.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.encoding_f, self.network_f = encoding_f, network_f
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        self._finite_difference_eps_f = None
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False, with_auxiliary_feature=False):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
                
                # print(self.encoding(points.view(-1, 3)).shape)
                # print(self.encoding_f(points.view(-1, 3)).shape)
                
                out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                out_f = self.network_f(self.encoding_f(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                
                sdf_c, feature_c = out[...,0], out
                sdf_f, feature_f = out_f[...,0], out_f
                sdf = sdf_c * 0.5 + sdf_f * 0.5
                feature = feature_c * 0.5 + feature_f * 0.5
                feature = torch.concat([feature, (points * 2 - 1)], dim=-1)
                
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  
                
                if with_laplace or with_auxiliary_feature:
                    eps = (self._finite_difference_eps + self._finite_difference_eps_f) / 2
                    rand_directions=torch.randn_like(points)
                    rand_directions=F.normalize(rand_directions,dim=-1)

                    #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
                    normals=F.normalize(grad,dim=-1)
                    tangent=torch.cross(normals, rand_directions)
                    rand_directions=tangent #set the random moving direction to be the tangent direction now
                    
                    points_shifted=points.clone()+rand_directions*eps

                if with_auxiliary_feature:
                    points_shifted_ = scale_anything(points_shifted, (-self.radius, self.radius), (0, 1))
                    points_shifted_feature = self.network(self.encoding(points_shifted_.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                    if 'feature_activation' in self.config:
                        points_shifted_feature = get_activation(self.config.feature_activation)(points_shifted_feature)
                if with_laplace:                    
                    offsets = torch.as_tensor(
                    [
                        [eps, 0.0, 0.0],
                        [-eps, 0.0, 0.0],
                        [0.0, eps, 0.0],
                        [0.0, -eps, 0.0],
                        [0.0, 0.0, eps],
                        [0.0, 0.0, -eps],
                    ]
                    ).to(points_)
                    points_shifted_d_ = (points_shifted[...,None,:] + offsets).clamp(-self.radius, self.radius)
                    points_shifted_d = scale_anything(points_shifted_d_, (-self.radius, self.radius), (0, 1))
                    points_shifted_d_sdf = self.network(self.encoding(points_shifted_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                    sdf_gradients_shifted = 0.5 * (points_shifted_d_sdf[..., 0::2] - points_shifted_d_sdf[..., 1::2]) / eps  

                    normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

                    dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
                    #the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
                    angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and pi when is opposite

                    laplace=angle/math.pi #map to [0,1 range]
                
                
        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            rv.append(laplace)
        if with_auxiliary_feature:
            rv.append(points_shifted_feature)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        sdf_c = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        sdf_f = self.network_f(self.encoding_f(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        # add in fine sdf
        sdf = sdf_c * 0.5 + sdf_f * 0.5
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        update_module_step(self.encoding_f, epoch, global_step)    
        update_module_step(self.network_f, epoch, global_step)  
        if isinstance(self.finite_difference_eps, float):
            self._finite_difference_eps = self.finite_difference_eps
        elif self.finite_difference_eps == 'progressive':
            hg_conf = self.config.xyz_encoding_config
            assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
            current_level = min(
                hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                hg_conf.n_levels
            )
            grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
            grid_size = 2 * self.config.radius / grid_res
            if grid_size != self._finite_difference_eps:
                rank_zero_info(f"Update finite_difference_eps to {grid_size}")
            self._finite_difference_eps = grid_size
            
            hg_conf_f = self.config.xyz_encoding_config_f
            assert hg_conf_f.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
            current_level_f = min(
                hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                hg_conf.n_levels
            )
            grid_res_f = hg_conf_f.base_resolution * hg_conf_f.per_level_scale**(current_level_f - 1)
            grid_size_f = 2 * self.config.radius / grid_res_f
            if grid_size_f != self._finite_difference_eps_f:
                rank_zero_info(f"Update finite_difference_eps_f to {grid_size_f}")
            self._finite_difference_eps_f = grid_size_f
        else:
            raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")
        
        
@models.register('volume-sdf-laplace-fusion-con')
# experimental version using sdf consistency, do not work quite well, and the training speed is low
class VolumeSDFLF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        encoding_f = get_encoding(3, self.config.xyz_encoding_config_f)
        network_f = get_mlp(encoding_f.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.encoding_f, self.network_f = encoding_f, network_f
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        self._finite_difference_eps_f = None
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False, with_auxiliary_feature=False, with_consistent=False):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
                
                # print(self.encoding(points.view(-1, 3)).shape)
                # print(self.encoding_f(points.view(-1, 3)).shape)
                
                out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                out_f = self.network_f(self.encoding_f(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                
                sdf_c, feature_c = out[...,0], out
                sdf_f, feature_f = out_f[...,0], out_f
                sdf = sdf_c * 0.5 + sdf_f * 0.5
                # sdf = torch.min(sdf_c, sdf_f)
                feature = feature_c * 0.5 + feature_f * 0.5
                feature = torch.concat([feature, (points * 2 - 1)], dim=-1)
                
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = (self._finite_difference_eps + self._finite_difference_eps_f) / 2
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        points_d_sdf_c = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        points_d_sdf_f = self.network_f(self.encoding_f(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        points_d_sdf = points_d_sdf_c * 0.5 + points_d_sdf_f * 0.5
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  
                
                if with_laplace or with_auxiliary_feature:
                    eps = (self._finite_difference_eps + self._finite_difference_eps_f) / 2
                    rand_directions=torch.randn_like(points)
                    rand_directions=F.normalize(rand_directions,dim=-1)

                    #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
                    normals=F.normalize(grad,dim=-1)
                    tangent=torch.cross(normals, rand_directions)
                    rand_directions=tangent #set the random moving direction to be the tangent direction now
                    
                    points_shifted=points.clone()+rand_directions*eps

                if with_auxiliary_feature:
                    points_shifted_ = scale_anything(points_shifted, (-self.radius, self.radius), (0, 1))
                    points_shifted_feature = self.network(self.encoding(points_shifted_.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                    if 'feature_activation' in self.config:
                        points_shifted_feature = get_activation(self.config.feature_activation)(points_shifted_feature)
                if with_laplace:                    
                    offsets = torch.as_tensor(
                    [
                        [eps, 0.0, 0.0],
                        [-eps, 0.0, 0.0],
                        [0.0, eps, 0.0],
                        [0.0, -eps, 0.0],
                        [0.0, 0.0, eps],
                        [0.0, 0.0, -eps],
                    ]
                    ).to(points_)
                    points_shifted_d_ = (points_shifted[...,None,:] + offsets).clamp(-self.radius, self.radius)
                    points_shifted_d = scale_anything(points_shifted_d_, (-self.radius, self.radius), (0, 1))
                    points_shifted_d_sdf = self.network(self.encoding(points_shifted_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                    sdf_gradients_shifted = 0.5 * (points_shifted_d_sdf[..., 0::2] - points_shifted_d_sdf[..., 1::2]) / eps  

                    normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

                    dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
                    #the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
                    angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and pi when is opposite

                    laplace=angle/math.pi #map to [0,1 range]
                
                consis_constraint = sdf
                # make sure testing stage
                if with_consistent and self.training:
                    sharp_val = 0.25
                    gradient_norm = F.normalize(grad, dim=-1)
                    points_moved = points + gradient_norm * sdf.unsqueeze(1)
                    points_moved = scale_anything(points_moved, (-self.radius, self.radius), (0, 1))
                    points_moved_ = points_moved
                    out_moved = self.network(self.encoding(points_moved.view(-1, 3))).view(*points_moved.shape[:-1], self.n_output_dims).float()
                    out_moved_f = self.network_f(self.encoding_f(points_moved.view(-1, 3))).view(*points_moved.shape[:-1], self.n_output_dims).float()
                    sdf_moved_c = out_moved[..., 0]
                    sdf_moved_f = out_moved_f[..., 0]
                    sdf_moved = sdf_moved_c * 0.5 + sdf_moved_f * 0.5
                    if 'sdf_activation' in self.config:
                        sdf_moved = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                    gradient_moved = torch.autograd.grad(
                            sdf_moved, points_moved_, grad_outputs=torch.ones_like(sdf_moved),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    gradient_moved_norm = F.normalize(gradient_moved, dim=-1)
                    consis_constraint = 1 - F.cosine_similarity(gradient_moved_norm, gradient_norm, dim=-1)
                    weight_moved = torch.exp(-sharp_val * torch.abs(sdf)).reshape(-1,consis_constraint.shape[-1]) 
                    consis_constraint = consis_constraint * weight_moved
                    

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            rv.append(laplace)
        if with_auxiliary_feature:
            rv.append(points_shifted_feature)
        if with_consistent:
            rv.append(consis_constraint)
            # add in loss
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        sdf_c = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        sdf_f = self.network_f(self.encoding_f(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        # add in fine sdf
        sdf = sdf_c * 0.5 + sdf_f * 0.5
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        update_module_step(self.encoding_f, epoch, global_step)    
        update_module_step(self.network_f, epoch, global_step)  
        if isinstance(self.finite_difference_eps, float):
            self._finite_difference_eps = self.finite_difference_eps
        elif self.finite_difference_eps == 'progressive':
            hg_conf = self.config.xyz_encoding_config
            assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
            current_level = min(
                hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                hg_conf.n_levels
            )
            grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
            grid_size = 2 * self.config.radius / grid_res
            if grid_size != self._finite_difference_eps:
                rank_zero_info(f"Update finite_difference_eps to {grid_size}")
            self._finite_difference_eps = grid_size
            
            hg_conf_f = self.config.xyz_encoding_config_f
            assert hg_conf_f.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
            current_level_f = min(
                hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                hg_conf.n_levels
            )
            grid_res_f = hg_conf_f.base_resolution * hg_conf_f.per_level_scale**(current_level_f - 1)
            grid_size_f = 2 * self.config.radius / grid_res_f
            if grid_size_f != self._finite_difference_eps_f:
                rank_zero_info(f"Update finite_difference_eps_f to {grid_size_f}")
            self._finite_difference_eps_f = grid_size_f
        else:
            raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")