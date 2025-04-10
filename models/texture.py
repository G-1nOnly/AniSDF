import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network
from systems.utils import update_module_step
from utils.misc import get_rank
from models.utils import Embedding


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        if self.config.use_appearance_embedding:
            print("Using appearance embedding!")
            self.n_input_dims += self.config.appearance_embedding_dim
            self.embedding_appearance = Embedding(self.config.max_imgs, self.config.appearance_embedding_dim)
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.encoding = encoding
        self.network = network
    
    def forward(self, features, dirs, camera_indices=None, ray_indices=None, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        
        # appearance embeddings
        if self.config.use_appearance_embedding:
            if self.training:
                assert camera_indices is not None, "Camera indices should be given during training when appearance embedding is used"
                appe_embd = self.embedding_appearance(camera_indices[ray_indices])
            elif camera_indices is not None and camera_indices.nelement() > 0:
                appe_embd = self.embedding_appearance(camera_indices).repeat(features.size()[0], 1)
            else:
                appe_embd = self.embedding_appearance.mean(dim=0).repeat(features.size()[0], 1)
                if not self.config.use_average_appearance_embedding:
                    appe_embd *= 0
            network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd, appe_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        else:
            network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)

        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}

@models.register('volume-radiance-uni')
# implementation of UniSDF (Instant versiomn)
class VolumeRadiance_Uni(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance_Uni, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + self.encoding.n_output_dims
        self.cam_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.ref_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.weight_network = get_mlp(self.config.input_feature_dim, 1, self.config.weight_network_config)
        
    def forward(self, features, viewdirs, normals):
        dirs = (viewdirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        refdirs = (refdirs + 1.) / 2. # (-1, 1) => (0, 1)
        refdirs_embd = self.encoding(refdirs.view(-1, self.n_dir_dims))

        network_inp = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1])], dim=-1)
        ref_weight = self.weight_network(network_inp)

        cam_network_inp = torch.cat([network_inp, dirs_embd], dim=-1)
        ref_network_inp = torch.cat([network_inp, refdirs_embd], dim=-1)
        cam_color = self.cam_network(cam_network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        ref_color = self.ref_network(ref_network_inp).view(*features.shape[:-1], self.n_output_dims).float()

        color = ref_weight * get_activation(self.config.color_activation)(ref_color) + \
                (1-ref_weight) * get_activation(self.config.color_activation)(cam_color)
                
        base = (1-ref_weight) * get_activation(self.config.color_activation)(cam_color)
        ref = ref_weight * get_activation(self.config.color_activation)(ref_color)
                        
        return color, base, ref
    
    def regularizations(self, out):
        return {}
        
@models.register('volume-radiance-ide')
# use IDE for the reflective direction, also try to model the emissive term
class VolumeRadiance_IDE(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance_IDE, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.encoding, self.encoding_ide = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + self.encoding.n_output_dims
        self.cam_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.ref_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.n_input_dims_ide = self.config.input_feature_dim + self.encoding_ide.n_output_dims
        self.light_network = get_mlp(self.n_input_dims_ide, self.n_output_dims, self.config.mlp_network_config)
        self.weight_network = get_mlp(self.config.input_feature_dim, 2, self.config.weight_network_config)
        self.kappa_inv = 0.64
        
    def forward(self, features, viewdirs, normals):
        dirs = (viewdirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        refdirs = (refdirs + 1.) / 2. # (-1, 1) => (0, 1)
        refdirs_embd = self.encoding(refdirs.view(-1, self.n_dir_dims))
    
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        roughness_act = nn.Softplus()
        roughness_act_scale = 1.0
        roughness_bias = -1
        if features.shape[0] != 0:
            kappa_inv = roughness_act_scale * roughness_act(features[..., -1] + roughness_bias)[0]
            self.kappa_inv = kappa_inv
        lightdirs_embd = self.encoding_ide(refdirs.view(-1, self.n_dir_dims), self.kappa_inv)
          
        network_inp = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1])], dim=-1)
        weight = self.weight_network(network_inp)
        
        cam_network_inp = torch.cat([network_inp, dirs_embd], dim=-1)
        ref_network_inp = torch.cat([network_inp, refdirs_embd], dim=-1)
        light_network_inp = torch.cat([network_inp, lightdirs_embd], dim=-1)
        cam_color = self.cam_network(cam_network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        ref_color = self.ref_network(ref_network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        light_color = self.light_network(light_network_inp).view(*features.shape[:-1], self.n_output_dims).float()

        ref_weight = weight[:, 0].unsqueeze(1)
        light_weight = weight[:, 1].unsqueeze(1)
        color = ref_weight * get_activation(self.config.color_activation)(ref_color) + \
                light_weight * get_activation(self.config.color_activation)(light_color) + \
                (1-ref_weight-light_weight) * get_activation(self.config.color_activation)(cam_color)
                
        base = (1-ref_weight-light_weight) * get_activation(self.config.color_activation)(cam_color)
        ref = ref_weight * get_activation(self.config.color_activation)(ref_color)
        light = light_weight * get_activation(self.config.color_activation)(light_color)
        
        return color, base, ref, light
    
    def regularizations(self, out):
        return {}
    
@models.register('volume-radiance-ASG')
class VolumeRadiance_ASG(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance_ASG, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + self.encoding.n_output_dims
        self.cam_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.weight_network = get_mlp(self.config.input_feature_dim, 1, self.config.weight_network_config)
        
        from .ASG_encoder import RenderingEquationEncoding
        self.num_theta = 8
        self.num_phi = 16
        self.num_asg = self.num_theta * self.num_phi
        self.ch_asg_feature = 128
        self.ch_per_theta = self.ch_asg_feature // self.num_theta
        self.ch_a = 2
        self.ch_la = 1
        self.ch_mu = 1
        self.ch_per_asg = self.ch_a + self.ch_la + self.ch_mu
        self.ch_normal_dot_viewdir = 1
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, device=get_rank())
        self.ASG_mlp = get_mlp(self.config.input_feature_dim, self.ch_asg_feature, self.config.ASG_network_config)
        self.asg_mlp = torch.nn.Sequential(torch.nn.Linear(self.ch_per_theta, self.num_phi * self.ch_per_asg)).to(get_rank())
        self.asg_color_mlp = get_mlp(self.num_asg * self.ch_a + self.ch_normal_dot_viewdir, self.n_output_dims, self.config.mlp_network_config)
        print("Using Anistropic Gaussians!")
            
    def asg_mlp_forward(self, asg_feature):
        asg_feature = asg_feature.view(-1, self.num_theta, self.num_phi)
        asg_params = self.asg_mlp(asg_feature)
        asg_params = asg_params.view(-1, self.num_theta, self.num_phi, self.ch_per_asg)
        
        a, la, mu = torch.split(asg_params, [self.ch_a, self.ch_la, self.ch_mu], dim=-1)
        return a, la, mu
    
    
    def forward(self, features, viewdirs, normals, positions=None):
        dirs = (viewdirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        # add in normalization
        refdirs = (refdirs + 1.) / 2. 
          
        network_inp = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1])], dim=-1)
        weight = self.weight_network(network_inp)
        
        cam_weight = weight[:, 0].unsqueeze(1)
        cam_network_inp = torch.cat([network_inp, dirs_embd], dim=-1)
        cam_color = self.cam_network(cam_network_inp).view(*features.shape[:-1], self.n_output_dims).float() 
        
        asg_input = network_inp
        # asg_input = torch.cat([network_inp, positions.view(-1, positions.shape[-1])], dim=-1)
        asg_fea = self.ASG_mlp(asg_input)
        a, la, mu = self.asg_mlp_forward(asg_fea)
        ree = self.ree_function(refdirs, a, la, mu) # N, num_theta, num_phi, ch_per_asg
        # ree = ree.view(ree.size(0), -1)
        ree = ree.view(-1, self.num_asg * self.ch_a)
        asg_color_inp = torch.cat([ree, VdotN], dim=-1)
        ASG_color = self.asg_color_mlp(asg_color_inp).float()
        
        color = cam_weight * get_activation(self.config.color_activation)(cam_color) + (1 - cam_weight) * ASG_color
        base = cam_weight * get_activation(self.config.color_activation)(cam_color)
        light = (1 - cam_weight) * ASG_color
        
        return color, base, light
    
    def regularizations(self, out):
        return {}


@models.register('volume-radiance-ASG-SG')
# use SG to model diffuse, works quite well for highly specular scenes
class VolumeRadiance_ASG_SG(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance_ASG_SG, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.specular_dim = self.config.get('specular_dim')
        self.sg_blob_num = self.config.get('sg_blob_num')
        self.cam_network = get_encoding_with_network(3, 9*self.sg_blob_num, self.config.sg_encoding_config, self.config.sg_network_config)
        self.weight_network = get_mlp(self.config.input_feature_dim, 1, self.config.weight_network_config)
        
        from .ASG_encoder import RenderingEquationEncoding
        self.num_theta = 8
        self.num_phi = 16
        self.num_asg = self.num_theta * self.num_phi
        self.ch_asg_feature = 128
        self.ch_per_theta = self.ch_asg_feature // self.num_theta
        self.ch_a = 2
        self.ch_la = 1
        self.ch_mu = 1
        self.ch_per_asg = self.ch_a + self.ch_la + self.ch_mu
        self.ch_normal_dot_viewdir = 1
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, device=get_rank())
        self.ASG_mlp = get_mlp(self.config.input_feature_dim, self.ch_asg_feature, self.config.ASG_network_config)
        self.asg_mlp = torch.nn.Sequential(torch.nn.Linear(self.ch_per_theta, self.num_phi * self.ch_per_asg)).to(get_rank())
        self.asg_color_mlp = get_mlp(self.num_asg * self.ch_a + self.ch_normal_dot_viewdir, self.n_output_dims, self.config.mlp_network_config)
        print("Using Anistropic Gaussians and spherical Gaussians!")
            
    def asg_mlp_forward(self, asg_feature):
        asg_feature = asg_feature.view(-1, self.num_theta, self.num_phi)
        asg_params = self.asg_mlp(asg_feature)
        asg_params = asg_params.view(-1, self.num_theta, self.num_phi, self.ch_per_asg)
        
        a, la, mu = torch.split(asg_params, [self.ch_a, self.ch_la, self.ch_mu], dim=-1)
        return a, la, mu
    
    def spherical_gaussian(self, viewdirs: torch.Tensor, lgtSGs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the specular component of a Spherical Gaussian (SG) model.

        Args:
            direction (torch.Tensor): The direction vector, in a shape of [N, 3]
            lgtSGs (torch.Tensor): The parameter of the SGs, in a shape of [N, sg_blob_num, 7]

        Returns:
            torch.Tensor: The specular component, in a shape of [N, 3]
        """
        
        viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
        
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True)) # (-1, 1), [N, sg_blob_num, 3]
        lgtSGMus = torch.sigmoid(lgtSGs[..., -3:])  # (0, 1), [N, sg_blob_num, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) #  positive values, [N, sg_blob_num, 3]
    
        specular = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        specular = torch.sum(specular, dim=-2)  # [..., 3]
        
        return specular
    
    def forward(self, features, viewdirs, normals, positions=None):
        dirs = (viewdirs + 1.) / 2. # (-1, 1) => (0, 1)
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        # add in normalization
        refdirs = (refdirs + 1.) / 2. 
          
        network_inp = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1])], dim=-1)
        weight = self.weight_network(network_inp)
        
        cam_weight = weight[:, 0].unsqueeze(1)
        cam_feature = self.cam_network(positions).reshape((-1, self.sg_blob_num, 9))
        cam_color = self.spherical_gaussian(dirs, cam_feature).float()       
        
        asg_input = network_inp
        asg_fea = self.ASG_mlp(asg_input)
        a, la, mu = self.asg_mlp_forward(asg_fea)
        ree = self.ree_function(refdirs, a, la, mu) # N, num_theta, num_phi, ch_per_asg
        ree = ree.view(-1, self.num_asg * self.ch_a)
        asg_color_inp = torch.cat([ree, VdotN], dim=-1)
        ASG_color = self.asg_color_mlp(asg_color_inp).float()
        
        color = cam_weight * cam_color + (1 - cam_weight) * ASG_color
        base = cam_weight * cam_color
        light = (1 - cam_weight) * ASG_color
        
        return color, base, light
    
    def regularizations(self, out):
        return {}
    
@models.register('volume-radiance-ASG-E')
# use appearance embeddings as in neuralangelo
class VolumeRadiance_ASG_E(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance_ASG_E, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + self.encoding.n_output_dims
        
        if self.config.use_appearance_embedding:
            print("Using appearance embeddings!")
            self.n_input_dims += self.config.appearance_embedding_dim
            self.embedding_appearance = Embedding(self.config.max_imgs, self.config.appearance_embedding_dim)
        
        self.cam_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.weight_network = get_mlp(self.config.input_feature_dim, 1, self.config.weight_network_config)
        
        from .ASG_encoder import RenderingEquationEncoding
        self.num_theta = 8
        self.num_phi = 16
        self.num_asg = self.num_theta * self.num_phi
        self.ch_asg_feature = 128
        self.ch_per_theta = self.ch_asg_feature // self.num_theta
        self.ch_a = 2
        self.ch_la = 1
        self.ch_mu = 1
        self.ch_per_asg = self.ch_a + self.ch_la + self.ch_mu
        self.ch_normal_dot_viewdir = 1
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, device=get_rank())
        self.ASG_mlp = get_mlp(self.config.input_feature_dim + self.config.appearance_embedding_dim, self.ch_asg_feature, self.config.ASG_network_config)
        self.asg_mlp = torch.nn.Sequential(torch.nn.Linear(self.ch_per_theta, self.num_phi * self.ch_per_asg)).to(get_rank())
        self.asg_color_mlp = get_mlp(self.num_asg * self.ch_a + self.ch_normal_dot_viewdir, self.n_output_dims, self.config.mlp_network_config)
        print("Using Anistropic Gaussians with Embeddings!")
            
    def asg_mlp_forward(self, asg_feature):
        asg_feature = asg_feature.view(-1, self.num_theta, self.num_phi)
        asg_params = self.asg_mlp(asg_feature)
        asg_params = asg_params.view(-1, self.num_theta, self.num_phi, self.ch_per_asg)
        
        a, la, mu = torch.split(asg_params, [self.ch_a, self.ch_la, self.ch_mu], dim=-1)
        return a, la, mu
    
    
    def forward(self, features, viewdirs, normals, positions=None, camera_indices=None, ray_indices=None):
        dirs = (viewdirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        # add in normalization
        refdirs = (refdirs + 1.) / 2. 
          
        # appearance embeddings
        if self.config.use_appearance_embedding:
            if self.training:
                assert camera_indices is not None, "Camera indices should be given during training when appearance embedding is used"
                appe_embd = self.embedding_appearance(camera_indices[ray_indices])
                print(appe_embd.shape)
            elif camera_indices is not None and camera_indices.nelement() > 0:
                appe_embd = self.embedding_appearance(camera_indices).repeat(features.size()[0], 1)
                print(appe_embd.shape)
            else:
                appe_embd = self.embedding_appearance.mean(dim=0).repeat(features.size()[0], 1)
                if not self.config.use_average_appearance_embedding:
                    appe_embd *= 0
                print(appe_embd.shape)
            print(appe_embd.shape)
            network_inp = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1]), appe_embd.view(-1, appe_embd.shape[-1])], dim=-1)
        else:
            network_inp = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1])], dim=-1)
        
        network_inp_w = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1])], dim=-1)
        weight = self.weight_network(network_inp_w)
        cam_weight = weight[:, 0].unsqueeze(1)
        cam_network_inp = torch.cat([network_inp, dirs_embd], dim=-1)
        cam_color = self.cam_network(cam_network_inp).view(*features.shape[:-1], self.n_output_dims).float() 
        
        asg_input = network_inp
        asg_fea = self.ASG_mlp(asg_input)
        a, la, mu = self.asg_mlp_forward(asg_fea)
        ree = self.ree_function(refdirs, a, la, mu) # N, num_theta, num_phi, ch_per_asg
        ree = ree.view(-1, self.num_asg * self.ch_a)
        asg_color_inp = torch.cat([ree, VdotN], dim=-1)
        ASG_color = self.asg_color_mlp(asg_color_inp).float()
        
        color = cam_weight * get_activation(self.config.color_activation)(cam_color) + (1 - cam_weight) * ASG_color
        base = cam_weight * get_activation(self.config.color_activation)(cam_color)
        light = (1 - cam_weight) * ASG_color
        
        return color, base, light
    
    def regularizations(self, out):
        return {}