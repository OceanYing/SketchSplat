import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import ipdb
import json

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
from gsplat import rasterization
from dacite import from_dict
from sklearn.neighbors import NearestNeighbors

from sketchgaussians.models.losses import MaskedL1Loss, WeightedL1Loss
from sketchgaussians.cameras.cameras import BaseCamera
from sketchgaussians.utils.misc_utils import unravel_index, random_quat_tensor, quats_to_rotmats_tensor, rotmats_to_quats_tensor
from sketchgaussians.utils.io_utils import write_gaussian_params_as_ply, read_pts_with_major_dirs_from_ply
from sketchgaussians.utils.eval_utils import sample_points_from_curves_diff
from sketchgaussians.utils.topology_utils import connect_endpts, merge_overlapping_strokes, merge_colinear_lines

@dataclass
class SketchSplattingConfig:

    if_duplicate_high_pos_grad: bool = True
    dup_threshold_type: str = "percentile"
    dup_threshold_value: float = 0.95
    dup_factor: int = 2
    dup_high_pos_grads_at_epoch: list = field(default_factory=lambda: [36, 46, 51, 76, 101, 126, 151])

    if_cull_low_opacity: bool = True
    cull_opacity_type: str = "absolute"
    cull_opacity_value: float = 0.05
    cull_opacity_at_epoch : list = field(default_factory=lambda: [80,160])

    if_cull_wayward: bool = True
    cull_wayward_method: str = "mean_distance"
    cull_wayward_num_neighbors: int = 10
    cull_wayward_threshold_type: str = "percentile_top"
    cull_wayward_threshold_value: float = 0.05
    cull_wayward_at_epoch : list = field(default_factory=lambda: [51,101,151])

    init_random_init: bool = False
    init_dup_rand_noise_scale: float = 0.05
    init_min_num_gaussians: int = 5000
    init_scales_type: str = "constant"
    init_scales_val: float = 0.005
    init_opacity_type: str = "constant"
    init_opacity_val: float = 0.08

    if_cull_gaussians_not_projecting : bool = True
    cull_gaussians_not_projecting_at_epoch : list = field(default_factory=lambda: [50,100,150])
    cull_gaussians_not_projecting_threshold : float = 0.35

    edge_detection_threshold: float = 0.5
    rasterize_mode = "antialiased"

    if_reset_opacity : bool = False
    reset_opacity_at_epoch : list = field(default_factory=lambda: [100])
    reset_opacity_value : float = 0.08

    stroke_json_path : str = None
    optimizable_opacity: bool = False
    sketch_add_noise: float = 0.0


'''
Much of the following class is inspired from the Splatfacto model in nerfstudio.
'''

class SketchSplatting(torch.nn.Module):

    def __init__(self, device = 'cuda'):
        self.device = device
        super().__init__()

    def poplutate_params(self, seed_points = None, viewcams = None, config = None):

        assert seed_points is not None, "Seed points need to be provided"
        assert viewcams is not None, "Viewcams need to be provided"
        assert config is not None, "Config needs to be provided"

        # import pdb
        # pdb.set_trace()

        config = from_dict(data_class=SketchSplattingConfig, data=config)
        self.config = config

        print("optimizable_opacity:", self.config.optimizable_opacity)

        self.seed_points = seed_points

        means = torch.nn.Parameter(self.seed_points)    
        constant_scale = torch.Tensor([config.init_scales_val ]).float()
        scales = torch.nn.Parameter(torch.log(constant_scale.repeat(means.shape[0], 3)))
        
        num_points = means.shape[0]
        self.viewcams = viewcams
        
        self.bg_pixels = []
        self.edge_pixels = []
        self.edge_masks = []
        
        self.crop_box = None # Can be used for cropping the gaussians to a specific region

        opacities = torch.nn.Parameter(torch.logit(config.init_opacity_val * torch.ones(num_points, 1)))
        quats = torch.nn.Parameter(random_quat_tensor(num_points))

        self.stroke_data_path = self.config.stroke_json_path
        
        self.noise_scale = self.config.sketch_add_noise

    def load_sketches(self):

        stroke_means, stroke_scales, stroke_opacities = self.load_and_init_strokes(data_path=self.stroke_data_path)

        self.gauss_params = torch.nn.ParameterDict(
            {
                "stroke_means": stroke_means,
                "stroke_scales": stroke_scales,
                "stroke_opacities": stroke_opacities,
            }
        )

        self.step = 0
    
    @property
    def num_points(self):
        return self.means.shape[0]
    
    @property
    def stroke_means(self):
        return self.gauss_params["stroke_means"]

    @property
    def stroke_scales(self):
        return self.gauss_params["stroke_scales"]
    
    @property
    def stroke_opacities(self):
        return self.gauss_params["stroke_opacities"]
    
    
    def get_gaussian_param_groups(self, param_name_list=None) -> Dict[str, List[torch.nn.Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        param_name_list = ["stroke_means", "stroke_scales", "stroke_opacities"] if param_name_list is None else param_name_list
        return_dict = {
            name: [self.gauss_params[name]]
            for name in param_name_list
        }
        return return_dict
            
    
    def load_and_init_strokes(self, data_path=None, noise_scale=0):
        
        # Open and load the JSON file
        with open(data_path, "r") as file:
            json_data = json.load(file)
        
        curve_paras = np.array(json_data["curves_ctl_pts"]).reshape(-1, 3)
        curves_ctl_pts = curve_paras.reshape(-1, 4, 3)
        lines_end_pts = np.array(json_data["lines_end_pts"]).reshape(-1, 2, 3)

        ## init strokes: as optimizable params
        self.num_curves = curves_ctl_pts.shape[0]
        self.num_lines = lines_end_pts.shape[0]

        stroke_means = torch.cat([  torch.Tensor(curves_ctl_pts).reshape(-1, 3), # (N1, 4, 3) Bezier curve
                                    torch.Tensor(lines_end_pts).reshape(-1, 3),  # (N2, 2, 3) Line curve
                                       ], dim=0)
        
        
        if self.noise_scale == 0:
            stroke_means = torch.nn.Parameter(stroke_means) # (N1*4 + N2*2, 3) Bezier curve
        elif self.noise_scale > 0:
            stroke_means = torch.nn.Parameter(stroke_means + torch.randn(stroke_means.shape) * self.noise_scale) # add white noise

        # initialize pts index of each strokes
        N1, N2 = (self.num_curves, self.num_lines)
        self.ptsidx = torch.arange(N1 * 4 + N2 * 2)    # (4 * N1 + 2 * N2)

        self.constant_scale = torch.Tensor([self.config.init_scales_val]).float()
        stroke_scales = torch.nn.Parameter(torch.log(self.constant_scale.repeat(N1+N2, 3)))

        stroke_opacities = torch.nn.Parameter(torch.logit(self.config.init_opacity_val * torch.ones(N1+N2, 1)))


        ### Load gaussians_filtered.ply
        orig_ply_filtered = os.path.join(os.path.dirname(data_path), "pts_with_major_dirs.ply")
        
        if not os.path.exists(orig_ply_filtered):
            print("no point cloud found as:", orig_ply_filtered)
            
            orig_ply_filtered = os.path.join(os.path.dirname(data_path), "udf_pointcloud_withdirection.ply")
            
            if not os.path.exists(orig_ply_filtered):
                print("no point cloud found as:", orig_ply_filtered)

        orig_pts, orig_dir = read_pts_with_major_dirs_from_ply(orig_ply_filtered)
        self.orig_pts = torch.tensor(orig_pts).float()
        self.orig_dir = torch.tensor(orig_dir).float()


        return stroke_means, stroke_scales, stroke_opacities

    def ptsidx_curve(self):
        assert self.ptsidx[:(4 * self.num_curves)].shape[0] == (self.num_curves*4), "in-consistent line stroke shape!"
        return self.ptsidx[:(4 * self.num_curves)]

    def ptsidx_line(self):
        assert self.ptsidx[(4 * self.num_curves):].shape[0] == (self.num_lines*2), "in-consistent line stroke shape!"
        return self.ptsidx[(4 * self.num_curves):]


    def sample_gs_from_strokes(self):
        # sample gaussian points evenly on strokes
        
        curve_means = self.gauss_params["stroke_means"][self.ptsidx_curve(), :].reshape(self.num_curves, 4, 3)
        line_means  = self.gauss_params["stroke_means"][self.ptsidx_line(),  :].reshape(self.num_lines, 2, 3)     # (N1, 2, 3) Line curve

        samp_pts_dir = sample_points_from_curves_diff(curve_means, line_means)
        samp_curve_points, samp_line_points, samp_curve_directions, samp_line_directions, samp_stroke_idx = samp_pts_dir
        samp_points = torch.cat([samp_curve_points, samp_line_points], dim=0)
        samp_dirs = torch.cat([samp_curve_directions, samp_line_directions], dim=0)
        samp_N = samp_points.shape[0]
        
        self.means = samp_points

        self.scales = self.stroke_scales[samp_stroke_idx, :]
        
        if self.config.optimizable_opacity is True:
            self.opacities = self.stroke_opacities[samp_stroke_idx, :]
        else:
            self.opacities = torch.ones(samp_N, device=samp_dirs.device)   # all saturated, may be changed

        self.samp_stroke_idx = samp_stroke_idx

        # Calculate quaternions with point direction of the stroke
        ref_dir = torch.tensor(np.array([1.0, 0.0, 0.0]), dtype=torch.float32, device=samp_dirs.device)
        r2 = torch.cross(samp_dirs, ref_dir[None, :])
        r2 = F.normalize(r2, p=2, dim=1)
        r3 = torch.cross(samp_dirs, r2)
        r3 = F.normalize(r3, p=2, dim=1)
        rot_mats = torch.stack([samp_dirs, r2, r3], dim=-1)
        self.quats = rotmats_to_quats_tensor(rot_mats)  # (N, 4)

        # write_pts_with_normals_as_ply(self.means.detach().cpu().numpy(), rot_mats[:, :, 0].detach().cpu().numpy(), "/fs/vulcan-projects/SER/SketchSplat/StrokeGaussians_v2/output/ABC/release_DexiNed/00000168_strokegs_stroke6_2/debug.ply")


    ### 1 ###
    def topo_connect_endpoints(self, optimizers, threshold=0.01):

        N1 = int(self.num_curves)
        N2 = int(self.num_lines)

        endidx = torch.cat([self.ptsidx[:N1*4].reshape(N1, 4)[:, [0,3]].flatten(),   # for curve, only take endpoints [0,3]
                            self.ptsidx[N1*4:]])

        endidx = torch.unique(endidx)   # remove repeated idx (since some strokes share the same endpoints)

        endpts = self.gauss_params["stroke_means"][endidx, :]    # get unique endpoints (NOTE: prevent repeated pts here)
        
        
        ### 1. run merge opereation
        merge_groups_loc = connect_endpts(endpts, threshold=threshold)

        if merge_groups_loc == {}:
            return

        ### 2. update idx list

        # NOTE: In merge_groups, the resulting indices are local indices (from 0 to endpts.shape[0]), we should change it to global idx using endidx
        # 2.1 transfer the dict from local to global idx, and get global idxlist that should be removed
        remove_ids = []
        merge_groups = {}
        for rep, indices in merge_groups_loc.items():
            rep_gl = endidx[rep].item()
            indices = endidx[indices].tolist()
            merge_groups[rep_gl] = indices
            remove_ids += indices

        # 2.2 Create a mapping from each value in the groups to the representative key.
        mapping = {}    
        for key, group in merge_groups.items():
            for val in group:
                mapping[val] = key
        # Replace each value in ptsidx with mapping[value] if it exists, otherwise keep the original value.
        ptsidx_update = torch.tensor([mapping.get(x, x) for x in self.ptsidx.tolist()])

        ## 2.3. make idx list continuous again (since some idx has been removed)
        _, ptsidx_new = torch.unique(ptsidx_update, sorted=True, return_inverse=True)

        # 2.4 remove pts from Gaussians
        # Then do the masking operation for pts using this global 'remove_ids' list
        remove_mask = torch.zeros(self.stroke_means.shape[0]).bool()
        remove_mask[remove_ids] = True

        # update pts index
        self.ptsidx = ptsidx_new
        param_name_list = ['stroke_means']
        self.cull_gaussians_stroke(optimizers, cull_mask=remove_mask, param_name_list=param_name_list, reset_rest=False)


        print("endpoint removed {} points".format(remove_mask.sum().item()))


    ### 2 ###
    def topo_merge_overlapping_sketch(self, optimizers, threshold=0.02, overlap_ratio=0.85):

        if self.means.shape[0] != self.samp_stroke_idx.shape[0]:    # make sure the number is the same
            self.sample_gs_from_strokes()

        delete_list = merge_overlapping_strokes(
            samp_pts_list=self.means, 
            samp_strokeid_list=self.samp_stroke_idx, 
            merge_threshold=threshold, 
            overlap_ratio=overlap_ratio)
        
        if len(delete_list) == 0:
            return
        remove_sids = torch.tensor(delete_list).long()
        num_sids = self.stroke_scales.shape[0]
        remove_sids_mask = torch.zeros(num_sids).bool()
        remove_sids_mask[remove_sids] = True

        self.topo_remove_sketch(optimizers, remove_sids_mask)
        
        print("overlap removed {} sketches".format(remove_sids_mask.sum().item()))


    ### 3 ###
    def topo_merge_colinear_sketch(self, optimizers, angle_thres=5, offset_tol=0.01, overlap_epsilon=0.01):
        
        # 1. find all connected lines edges (only lines considered)

        # 2. evaluate if the lines are colinear

        # 3. delete lines and add new lines

        N1 = int(self.num_curves)
        N2 = int(self.num_lines)

        means_all = self.gauss_params["stroke_means"].clone()
        pidx_c = self.ptsidx[:N1*4].reshape(N1, 4).clone()
        pidx_l = self.ptsidx[N1*4:].reshape(N2, 2).clone()

        merge_dict, pidx_new2 = merge_colinear_lines(
            points=means_all.clone(), 
            idxlist=pidx_l.clone(),
            angle_thresh=angle_thres,
            offset_tol=offset_tol,
            overlap_epsilon=overlap_epsilon
            )
        
        ### gather lines to be removed
        parent_sids = []
        remove_sids = []
        for rep, indices in merge_dict.items():
            parent_sids.append(rep)
            remove_sids += indices
        if len(remove_sids) == 0:
            return
        remove_sids = torch.tensor(remove_sids).long() + N1     # local to global (by adding N1)
        
        parent_sids = torch.tensor(parent_sids).long()
        pidx_l[parent_sids, :] = pidx_new2[parent_sids, :]  # update with unioned edges

        num_sids = self.stroke_scales.shape[0]
        remove_sids_mask = torch.zeros(num_sids).bool()
        remove_sids_mask[remove_sids] = True
        maintain_sids = torch.arange(num_sids)[~remove_sids_mask]

        pidx_c_new = pidx_c[maintain_sids[maintain_sids < N1], :]
        pidx_l_new = pidx_l[maintain_sids[maintain_sids >= N1] - N1, :]

        pidx_new = torch.cat([pidx_c_new.flatten(), pidx_l_new.flatten()])

        remove_pids = self.ptsidx[~torch.isin(self.ptsidx, pidx_new)]
        remove_pids_mask = torch.zeros(self.stroke_means.shape[0]).bool()
        remove_pids_mask[remove_pids] = True

        ### update self.ptsidx, stroke_means, stroke_scales, number

        param_name_list = ['stroke_means']
        self.cull_gaussians_stroke(optimizers, cull_mask=remove_pids_mask, param_name_list=param_name_list, reset_rest=False)
        param_name_list = ['stroke_scales']
        self.cull_gaussians_stroke(optimizers, cull_mask=remove_sids_mask, param_name_list=param_name_list, reset_rest=False)
        param_name_list = ['stroke_opacities']
        self.cull_gaussians_stroke(optimizers, cull_mask=remove_sids_mask, param_name_list=param_name_list, reset_rest=False)

        _, ptsidx_new_contiguous = torch.unique(pidx_new, sorted=True, return_inverse=True)

        self.ptsidx = ptsidx_new_contiguous.clone()
        self.num_curves -= (remove_sids < N1).sum()
        self.num_lines -= (remove_sids >= N1).sum()


        print("colinear removed {} sketches".format(remove_sids_mask.sum().item()))

        # pass


    ### 4 ###
    def topo_add_new_sketch(self, optimizers, max_add_num=10, max_add_ratio=0.2, new_length=0.06):

        # 1. evaluate current point coverage
        # 2. randomly choose some points to add
        # 3. update optimizer

        # params: max_add_num=20, max_add_ratio=0.2, dist = 

        # with torch.no_grad():
        
        if self.means.shape[0] != self.samp_stroke_idx.shape[0]:    # make sure the number is the same
            self.sample_gs_from_strokes()
        
        # find B's KNN, if in A?
        A = self.means.clone()
        B = self.orig_pts.clone().to(A.device)
        Bdir = self.orig_dir.clone().to(A.device)

        AB = torch.cat([A, B], dim=0)

        dists, inds = self.k_nearest_sklearn(AB.detach(), k=5)

        uncover_mask = torch.from_numpy((inds[A.shape[0]:, :] < A.shape[0]).sum(axis=1) < 1)
        uncover_num = uncover_mask.sum()

        if uncover_num == 0:
            return
        
        add_num = min(max_add_num, int(max_add_ratio * uncover_num))

        choose_idx = torch.randperm(uncover_num)[:add_num]
        choose_idx

        new_pts = B[uncover_mask, :][choose_idx, :]
        new_dir = Bdir[uncover_mask, :][choose_idx, :]

        Nc = add_num // 2   # add curve number
        Nl = add_num - Nc   # add line number

        new_cs = torch.stack([
            new_pts - (new_length/2) * new_dir,
            new_pts - (new_length/6) * new_dir,
            new_pts + (new_length/6) * new_dir,
            new_pts + (new_length/2) * new_dir
        ], dim=1)[:Nc, :, :].reshape(-1, 3)

        new_ls = torch.stack([
            new_pts - (new_length/2) * new_dir,
            new_pts + (new_length/2) * new_dir
        ], dim=1)[Nc:, :, :].reshape(-1, 3)

        new_pts_all = torch.cat([new_cs, new_ls], dim=0)
        stroke_means = torch.cat([self.stroke_means, new_pts_all], dim=0)
        num_old = self.stroke_means.shape[0]

        N1 = int(self.num_curves)
        N2 = int(self.num_lines)
        
        pidx_c = self.ptsidx[:N1*4].clone()
        pidx_l = self.ptsidx[N1*4:].clone()


        self.ptsidx = torch.cat([
            pidx_c, num_old + torch.arange(Nc*4), 
            pidx_l, num_old + torch.arange(Nc*4, Nc*4 + Nl*2)
        ])

        self.num_curves += Nc
        self.num_lines += Nl

        new_scales_all = torch.log(self.constant_scale.repeat(Nc+Nl, 3)).to(A.device)
        stroke_scales = torch.cat([
            self.stroke_scales[:N1, :], new_scales_all[:Nc, :], 
            self.stroke_scales[N1:, :], new_scales_all[Nc:, :]
            ], dim=0)
        
        new_opacities_all = torch.logit(self.config.init_opacity_val * torch.ones(Nc+Nl, 1)).to(A.device)
        stroke_opacities = torch.cat([
            self.stroke_opacities[:N1, :], new_opacities_all[:Nc, :], 
            self.stroke_opacities[N1:, :], new_opacities_all[Nc:, :]
            ], dim=0)
        
        self.dup_gaussians_stroke(optimizers, dup_num=(Nc*4 + Nl*2), new_param=stroke_means, param_name='stroke_means')
        
        self.dup_gaussians_stroke(optimizers, dup_num=[Nc, Nl], new_param=stroke_scales, param_name='stroke_scales', split=[N1,N2])
        self.dup_gaussians_stroke(optimizers, dup_num=[Nc, Nl], new_param=stroke_opacities, param_name='stroke_opacities', split=[N1,N2])


    ### 4.2: transfer ###
    def topo_transfer_curve_to_line(self, optimizers, mode='control', threshold=0.95):

        """
        mode 1: 'control': use control pts to calculate linearity
        mode 2: 'sample': use sampled pts to calculate linearity (heavier)
        """
        
        N1 = int(self.num_curves)
        N2 = int(self.num_lines)

        if N1 == 0:     # No curve exists
            return

        ### all bezier curve
        curve_pids = self.ptsidx[:N1*4].reshape(N1, 4)   # (N1, 4)
        curve_pts = self.gauss_params["stroke_means"][curve_pids, :].clone()    # (N1, 4, 3)
        curve_scale = self.gauss_params["stroke_scales"][:N1, :].clone()    # (N1, 3)
        curve_opacity = self.gauss_params["stroke_opacities"][:N1, :].clone()    # (N1, 1)

        curve_linearity = self.colinearity_score_batch(curve_pts)   # (N1,)

        if (curve_linearity > threshold).sum() == 0:
            return
        
        remove_sids = torch.argwhere(curve_linearity > threshold)[0] # curve idx here is also global sid, since curve first
        
        Nc = 0
        Nl = len(remove_sids)
        
        new_pts = curve_pts[remove_sids][:,[0,3],:]    # (Nl, 2, 3)

        new_ls = new_pts.reshape(-1, 3)

        new_pts_all = new_ls
        stroke_means = torch.cat([self.stroke_means, new_pts_all], dim=0)
        num_old = self.stroke_means.shape[0]
        
        pidx_c = self.ptsidx[:N1*4].clone()
        pidx_l = self.ptsidx[N1*4:].clone()


        self.ptsidx = torch.cat([
            pidx_c, 
            pidx_l, num_old + torch.arange(Nl*2)
        ])

        self.num_curves += 0
        self.num_lines += Nl

        # new_scales_all = torch.log(self.constant_scale.repeat(Nc+Nl, 3)).to(A.device)
        new_scales_all = curve_scale[remove_sids, :].clone()
        stroke_scales = torch.cat([
            self.stroke_scales[:N1, :], 
            self.stroke_scales[N1:, :], new_scales_all
            ], dim=0)
        
        # new_opacities_all = torch.logit(self.config.init_opacity_val * torch.ones(Nc+Nl, 1)).to(A.device)
        new_opacities_all = curve_opacity[remove_sids, :].clone()
        stroke_opacities = torch.cat([
            self.stroke_opacities[:N1, :], 
            self.stroke_opacities[N1:, :], new_opacities_all
            ], dim=0)

        # import pdb
        # pdb.set_trace()
        self.dup_gaussians_stroke(optimizers, dup_num=(Nc*4 + Nl*2), new_param=stroke_means, param_name='stroke_means')
        self.dup_gaussians_stroke(optimizers, dup_num=[Nc, Nl], new_param=stroke_scales, param_name='stroke_scales', split=[N1,N2])
        self.dup_gaussians_stroke(optimizers, dup_num=[Nc, Nl], new_param=stroke_opacities, param_name='stroke_opacities', split=[N1,N2])

        # import pdb
        # pdb.set_trace()

        ### remove curve sketch
        # remove_sids = torch.tensor(delete_list).long()
        num_sids = self.stroke_scales.shape[0]
        remove_sids_mask = torch.zeros(num_sids).bool()
        remove_sids_mask[remove_sids] = True
        self.topo_remove_sketch(optimizers, remove_sids_mask)

        # import pdb
        # pdb.set_trace()

        # self.num_curves -= Nl
        # self.num_lines -= 0

        print("Transfer {} curves to lines".format(Nl))


    ### 5 ###
    def topo_filter_low_opacity(self, optimizers):
        
        sigm_opacities = torch.sigmoid(self.stroke_opacities)
        # print("min max opacity:", sigm_opacities.min().item(), sigm_opacities.max().item())

        remove_sids_mask = (sigm_opacities < self.config.cull_opacity_value).squeeze()

        if remove_sids_mask.sum() == 0:
            return

        self.topo_remove_sketch(optimizers, remove_sids_mask)
        
        print("low_opacity removed {} sketches".format(remove_sids_mask.sum().item()))


    ### 6 ###
    def topo_filter_not_projecting(self, optimizers, min_projecting_fraction = 0.1, bad_retio_tol=0.7):

        self.sample_gs_from_strokes()
        
        num_gs = self.means.shape[0]
        num_frames = len(self.viewcams)

        gs_visib_matrix = torch.zeros(num_gs, num_frames, dtype=torch.bool)
        
        for idx, viewcam in enumerate(self.viewcams):

            P = viewcam.K.cpu() @ viewcam.viewmat.cpu()[:3, :4]
            w, h = viewcam.width, viewcam.height
            gaussian_means_h = torch.cat([self.means.detach().cpu(), torch.ones(num_gs, 1)], dim=-1)
            projected_means = torch.matmul(P, gaussian_means_h.t()).t()
            projected_means = projected_means[:, :2] / projected_means[:, 2:]
            projected_means_r = projected_means.round().long()
            good_inds = (projected_means_r[:, 0] >= 0) & (projected_means_r[:, 0] < w) & (projected_means_r[:, 1] >= 0) & (projected_means_r[:, 1] < h)
            projecting_within = projected_means_r[good_inds]
            projecting_on_edge = self.edge_masks[idx][projecting_within[:, 1], projecting_within[:, 0]]
            gs_visib_matrix[good_inds, idx] = projecting_on_edge
        
        mean_projections = torch.mean(gs_visib_matrix.float(), dim=1)
        cull_mask = mean_projections < min_projecting_fraction

        unique_sids = torch.unique(self.samp_stroke_idx)
        remove_sids_mask = torch.zeros(self.num_curves+self.num_lines).bool()
        for sid in unique_sids:
            sid_mask = self.samp_stroke_idx == sid
            bad_ratio = cull_mask[sid_mask].sum() / sid_mask.sum()
            remove_sids_mask[sid] = bad_ratio > bad_retio_tol     # more than 70% of the edge cannot be seen
        
        if remove_sids_mask.sum() == 0:
            return
        else:
            print("proj_vis removed {} sketches".format(remove_sids_mask.sum().item()))

        self.topo_remove_sketch(optimizers, remove_sids_mask)


    ### 6.2 ###
    def topo_filter_not_projecting_noise(self, optimizers, min_projecting_fraction = 0.1, bad_retio_tol=0.7):

        self.sample_gs_from_strokes()
        
        num_gs = self.means.shape[0]
        num_frames = len(self.viewcams)

        gs_visib_matrix = torch.zeros(num_gs, num_frames, dtype=torch.bool)
        
        for idx, viewcam in enumerate(self.viewcams):

            P = viewcam.K.cpu() @ viewcam.viewmat.cpu()[:3, :4]
            w, h = viewcam.width, viewcam.height
            gaussian_means_h = torch.cat([self.means.detach().cpu(), torch.ones(num_gs, 1)], dim=-1)
            projected_means = torch.matmul(P, gaussian_means_h.t()).t()
            projected_means = projected_means[:, :2] / projected_means[:, 2:]
            projected_means_r = projected_means.round().long()
            good_inds = (projected_means_r[:, 0] >= 0) & (projected_means_r[:, 0] < w) & (projected_means_r[:, 1] >= 0) & (projected_means_r[:, 1] < h)
            projecting_within = projected_means_r[good_inds]
            projecting_on_edge = self.edge_masks[idx][projecting_within[:, 1], projecting_within[:, 0]]
            gs_visib_matrix[good_inds, idx] = projecting_on_edge
        
        mean_projections = torch.mean(gs_visib_matrix.float(), dim=1)
        # ipdb.set_trace()
        cull_mask = mean_projections < min_projecting_fraction

        unique_sids = torch.unique(self.samp_stroke_idx)
        remove_sids_mask = torch.zeros(self.num_curves+self.num_lines).bool()
        for sid in unique_sids:
            sid_mask = self.samp_stroke_idx == sid
            bad_pts = cull_mask[sid_mask].sum()
            remove_sids_mask[sid] = (bad_pts == sid_mask.sum())      # all sampled points are bad
        
        if remove_sids_mask.sum() == 0:
            return
        else:
            print("proj_vis removed {} sketches".format(remove_sids_mask.sum().item()))

        self.topo_remove_sketch(optimizers, remove_sids_mask)


    def topo_remove_sketch(self, optimizers, remove_sids_mask):

        num_sids = self.stroke_scales.shape[0]
        remove_sids = torch.arange(num_sids)[remove_sids_mask.cpu()]
        maintain_sids = torch.arange(num_sids)[~remove_sids_mask.cpu()]

        N1 = int(self.num_curves)
        N2 = int(self.num_lines)

        pidx_c = self.ptsidx[:N1*4].reshape(N1, 4).clone()
        pidx_l = self.ptsidx[N1*4:].reshape(N2, 2).clone()

        pidx_c_new = pidx_c[maintain_sids[maintain_sids < N1], :]
        pidx_l_new = pidx_l[maintain_sids[maintain_sids >= N1] - N1, :]

        pidx_new = torch.cat([pidx_c_new.flatten(), pidx_l_new.flatten()])

        remove_pids = self.ptsidx[~torch.isin(self.ptsidx, pidx_new)]
        remove_pids_mask = torch.zeros(self.stroke_means.shape[0]).bool()
        remove_pids_mask[remove_pids] = True

        param_name_list = ['stroke_means']
        self.cull_gaussians_stroke(optimizers, cull_mask=remove_pids_mask, param_name_list=param_name_list, reset_rest=False)
        param_name_list = ['stroke_scales']
        self.cull_gaussians_stroke(optimizers, cull_mask=remove_sids_mask, param_name_list=param_name_list, reset_rest=False)
        param_name_list = ['stroke_opacities']
        self.cull_gaussians_stroke(optimizers, cull_mask=remove_sids_mask, param_name_list=param_name_list, reset_rest=False)

        _, ptsidx_new_contiguous = torch.unique(pidx_new, sorted=True, return_inverse=True)
        
        self.ptsidx = ptsidx_new_contiguous.clone()
        self.num_curves -= (remove_sids < N1).sum()
        self.num_lines -= (remove_sids >= N1).sum()


    def cull_gaussians_stroke(self, optimizers, cull_mask, param_name_list=None, reset_rest=False):
        for name, param in self.gauss_params.items():
            if (param_name_list is None) or (name in param_name_list):
                self.gauss_params[name] = param[~cull_mask]
        
        if reset_rest:
            self.reset_opacities()

        self.remove_from_all_optim(optimizers, cull_mask, param_name_list=param_name_list)
        
        # self.absgrads = self.absgrads[~cull_mask]
        
        num_culled = torch.sum(cull_mask).item()
        # print(f"Culled {num_culled} control pts")
    

    def colinearity_score_batch(self, pts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute a colinearity score for a batch of point-sets.

        Args:
            pts: Tensor of shape (..., N, D), where:
                - ... represents any number of batch dimensions (e.g., B)
                - N is the number of points per set (e.g., 4)
                - D is the dimensionality of each point (e.g., 2 or 3)
            eps: Small constant to avoid division by zero.

        Returns:
            Tensor of shape (...) giving a colinearity score in [0, 1] for each set:
            - 1.0 means the points lie exactly on a line.
            - Lower values indicate more deviation from perfect colinearity.
        """
        # 1) Center each set of points along the N dimension
        X = pts - pts.mean(dim=-2, keepdim=True)          # shape (..., N, D)

        # 2) Compute singular values for each batch-element
        #    torch.linalg.svdvals handles batched inputs automatically.
        S = torch.linalg.svdvals(X)                      # shape (..., min(N, D))

        # 3) Score = largest singular value / sum of all singular values
        score = (S[..., 0] / (S.sum(dim=-1) + eps)).clamp(0.0, 1.0)  # shape (...)

        return score

    
    # required for enforcing the geometric constraints
    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    
    # Functions required for the occlusion aware projection loss - is slow and can be optimized
    def compute_image_masks(self, gt_images):

        for image in gt_images:
            # print("Computing gt image shape ", image.shape)
            edge_mask = image >= self.config.edge_detection_threshold
            self.edge_masks.append(edge_mask)

        print("Computed masks for all images")

    def sample_pixels_for_loss(self, image_idx, ratio_edge_to_bg:float = 1):
        """
        Sample pixels from the image for computing the loss
        """
        bg_pixels = self.bg_pixels[image_idx]
        edge_pixels = self.edge_pixels[image_idx]

        num_bg_pixels = int(ratio_edge_to_bg * len(edge_pixels))
        bg_pixels = bg_pixels[torch.randperm(len(bg_pixels))[:num_bg_pixels]]
    

        return edge_pixels, bg_pixels
    
    # Weighted L1 loss for the occlusion aware projection loss
    def compute_weight_masks(self):
        assert hasattr(self, "edge_masks"), "Edge masks need to be computed first"
        assert self.edge_masks is not None, "Edge masks need to be computed first"

        self.weight_masks = []
        for edge_mask in self.edge_masks:
            
            num_edge_pixels = edge_mask.sum()
            num_bg_pixels = (~edge_mask).sum()
            
            edge_weight = num_bg_pixels / (num_edge_pixels + num_bg_pixels)
            bg_weight = num_edge_pixels / (num_edge_pixels + num_bg_pixels)

            weight_mask = torch.zeros_like(edge_mask, dtype = torch.float)
            weight_mask[edge_mask] = edge_weight
            weight_mask[~edge_mask] = bg_weight
            self.weight_masks.append(weight_mask)


    # Functions required for obtaining the rendered image from a single view - should be batched for faster training
    def get_outputs(self, camera: BaseCamera) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]

            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            quats_crop = quats_crop / torch.norm(quats_crop, dim=-1, keepdim=True)
        else:
            opacities_crop = self.opacities
            means_crop = self.means

            scales_crop = self.scales
            quats_crop = self.quats


        BLOCK_WIDTH = 16

        viewmat = camera.get_viewmat()
        K = camera.get_K()

        # W, H = int(camera.width.item()), int(camera.height.item())
        W, H = camera.width, camera.height
        self.last_size = (H, W)

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        
        render_mode = "RGB"
        colors_crop = torch.ones(means_crop.shape[0], 3).cuda()

        # ipdb.set_trace()
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )

        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()

        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        rgb = render[:, ..., :3]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        depth_im = None

        return {
            "rgb": rgb.squeeze(0),  
            "depth": depth_im, 
            "accumulation": alpha.squeeze(0),
        }
    
    def compute_projection_loss(self, output_image, gt_image, image_index = None, strategy = "bg_edge_ratio", bg_edge_pixel_ratio = 1.0, loss_type: str = "l1"):

        if strategy == "whole":
            if loss_type == "l1":
                criterion = torch.nn.functional.l1_loss
            elif loss_type == "l2":
                criterion = torch.nn.functional.mse_loss
            loss = criterion(output_image, gt_image)
            return loss

        elif strategy == "bg_edge_ratio":
            masked_l1_loss = MaskedL1Loss()
            edge_loss = masked_l1_loss(output_image, gt_image, self.edge_masks[image_index])
            
            # sample pixels from bg
            num_bg_pixels = int(bg_edge_pixel_ratio * self.edge_masks[image_index].sum())
            bg_mask = ~self.edge_masks[image_index]
            bg_mask_1 = torch.where(bg_mask)[0]
            bg_flat_select_1 = torch.randperm(len(bg_mask_1))[:num_bg_pixels]
            indices = unravel_index(bg_flat_select_1, bg_mask.shape)

            bg_mask_final = torch.zeros_like(bg_mask, dtype = torch.bool)
            bg_mask_final[indices[:,0], indices[:,1]] = True
            
            # compute loss for edges and sample bg
            bg_loss = masked_l1_loss(output_image, gt_image, bg_mask_final)
            loss = edge_loss + bg_loss

        elif strategy == "weighted":
            weighted_l1_loss = WeightedL1Loss()
            weight_mask = self.weight_masks[image_index]
            loss = weighted_l1_loss(output_image, gt_image, weight_mask)
        
        else:
            raise ValueError(f"Unknown projection loss strategy: {strategy}")

        return loss

    def update_nearest_neighbors(self):
        # compute the nearest neighbors for each point
        k = self.dir_loss_num_nn
        # check for nan values in the means and replace them with 0
        points = self.means.data
        if torch.isnan(points).sum() > 0:
            points[torch.isnan(points)] = 0
            print("Points with nan values ", torch.isnan(points).sum())

        start_time = time.time()
        if self.dir_loss_enforce_method != 'enforce_half':
            _, indices = self.k_nearest_sklearn(points, k+1)
            
        elif self.dir_loss_enforce_method == 'enforce_half':
            _, indices = self.k_nearest_sklearn(points, 2*k+1)
        
        end_time = time.time()
        # print(f"Time taken to compute nearest neighbors {end_time - start_time}")
        self.nn_indices = indices[:,1:]

    def compute_direction_loss(self, visualize = False):
        
        k = self.dir_loss_num_nn
        inds = torch.from_numpy(self.nn_indices).long()
        
        # get the major dorections for each gaussian
        rotmats = quats_to_rotmats_tensor(self.quats)

        ### op1: find max scale
        # scales = torch.exp(self.scales)
        # argmax_scales = torch.argmax(torch.abs(scales), dim=-1)
        # rotmats = rotmats.to(self.device)
        # major_dirs = rotmats[torch.arange(self.num_points), :, argmax_scales]

        ### op2: use the first scale
        major_dirs = rotmats[torch.arange(self.num_points), :, 0]   # NOTE: we have set the first scale axis as the major axis

        # get the directions towards the nearest neighbors
        neighbor_dirs = self.means[:, None, :] - self.means[inds]
        neighbor_dirs = neighbor_dirs / (torch.norm(neighbor_dirs, dim=-1, keepdim=True) + 1e-6)

        if self.dir_loss_enforce_method != 'enforce_half':
            alignment = torch.abs(torch.sum(major_dirs[:, None, :] * neighbor_dirs, dim=-1))
            mean_alignment = torch.mean(alignment, dim=-1)

        elif self.dir_loss_enforce_method == 'enforce_half':
            alignment = torch.abs(torch.sum(major_dirs[:, None, :] * neighbor_dirs, dim=-1))
            alignment_sorted, _ = torch.sort(alignment, dim=-1, descending=True)
            mean_alignment = torch.mean(alignment_sorted[:,:k], dim=-1)

        loss = 1.0 - torch.mean(mean_alignment)

        return loss

    def compute_ratio_loss(self):
        # get the ratio second largest to the largest scale for each gaussian
        scales = torch.exp(self.scales)
        sorted_scales, _ = torch.sort(scales, dim=-1, descending=True)
        ratio = sorted_scales[:, 1] / sorted_scales[:, 0]
        return torch.mean(ratio)
    
    
    ## Culling and duplication ##
    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state


    def remove_from_all_optim(self, optimizers, deleted_mask, param_name_list=None):
        param_groups = self.get_gaussian_param_groups(param_name_list=param_name_list)
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()


    def cull_gaussians(self, optimizers, cull_mask, reset_rest = True):
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = param[~cull_mask]
        
        if reset_rest:
            self.reset_opacities()

        self.remove_from_all_optim(optimizers, cull_mask)
        
        num_culled = torch.sum(cull_mask).item()
        print(f"Culled {num_culled} gaussians")
    
    def reset_opacities(self):
        self.stroke_opacities.data = torch.clamp(
                    self.stroke_opacities.data,
                    max=self.config.reset_opacity_value
                )

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            
            dup_exp_avg_list = [torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims) for i in range(self.config.dup_factor-1)]
            dup_exp_avg_sq_list = [torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims) for i in range(self.config.dup_factor-1)]

            param_state["exp_avg"] = torch.cat(
                [param_state["exp_avg"]] + dup_exp_avg_list,
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [param_state["exp_avg_sq"]] + dup_exp_avg_sq_list,
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n = 1, param_name_list=None):
        param_groups = self.get_gaussian_param_groups(param_name_list=param_name_list)
            
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers[group], dup_mask, param, n)

    
    def dup_gaussians(self, optimizers, dup_mask):
        for name, param in self.gauss_params.items():
            if name == "means":
                # add small gaussian noise to the duplicated points
                dup_means_list = [param[dup_mask] for i in range(self.config.dup_factor-1)]
                dup_means_tensor = torch.cat(dup_means_list, dim=0)
                dup_means_tensor += torch.randn_like(dup_means_tensor) * self.config.init_dup_rand_noise_scale
                self.gauss_params[name] = torch.cat([param, dup_means_tensor],  dim=0)
            else:
                concat_list = [param] + [param[dup_mask] for i in range(self.config.dup_factor-1)]
                self.gauss_params[name] = torch.cat(concat_list, dim=0)

        self.dup_in_all_optim(optimizers, dup_mask, n=1)
        num_dup = torch.sum(dup_mask).item()
        print(f"Duplicated {num_dup} gaussians")



    def dup_in_optim_stroke(self, optimizer, dup_num, new_params, n=2, split=None):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))

            if split is None:
                dup_exp_avg_list = [torch.zeros_like(param_state["exp_avg"][torch.zeros(dup_num).long()]).repeat(*repeat_dims) for i in range(1)]
                dup_exp_avg_sq_list = [torch.zeros_like(param_state["exp_avg_sq"][torch.zeros(dup_num).long()]).repeat(*repeat_dims) for i in range(1)]
                param_state["exp_avg"] = torch.cat(
                    [param_state["exp_avg"]] + dup_exp_avg_list,
                    dim=0,
                )
                param_state["exp_avg_sq"] = torch.cat(
                    [param_state["exp_avg_sq"]] + dup_exp_avg_sq_list,
                    dim=0,
                )
            else:
                dup_exp_avg_list_1 = [torch.zeros_like(param_state["exp_avg"][torch.zeros(dup_num[0]).long()]).repeat(*repeat_dims) for i in range(1)]
                dup_exp_avg_list_2 = [torch.zeros_like(param_state["exp_avg"][torch.zeros(dup_num[1]).long()]).repeat(*repeat_dims) for i in range(1)]
                dup_exp_avg_sq_list_1 = [torch.zeros_like(param_state["exp_avg_sq"][torch.zeros(dup_num[0]).long()]).repeat(*repeat_dims) for i in range(1)]
                dup_exp_avg_sq_list_2 = [torch.zeros_like(param_state["exp_avg_sq"][torch.zeros(dup_num[1]).long()]).repeat(*repeat_dims) for i in range(1)]
                param_state["exp_avg"] = torch.cat([param_state["exp_avg"][:split[0], ...]] + dup_exp_avg_list_1 + 
                                                   [param_state["exp_avg"][split[0]:, ...]] + dup_exp_avg_list_2, dim=0)
                param_state["exp_avg_sq"] = torch.cat([param_state["exp_avg_sq"][:split[0], ...]] + dup_exp_avg_sq_list_1 + 
                                                      [param_state["exp_avg_sq"][split[0]:, ...]] + dup_exp_avg_sq_list_2, dim=0)

        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_gaussians_stroke(self, optimizers, dup_num, new_param, param_name, split=None):
        self.gauss_params[param_name] = new_param

        self.dup_in_optim_stroke(optimizers[param_name], dup_num, [self.gauss_params[param_name]], n=1, split=split)

        print(f"Duplicated {torch.sum(torch.tensor(dup_num)).item()} gaussians")



    
    def cull_gaussians_opacity(self, optimizers):
        
        if self.config.cull_opacity_type == "percentile":
            cull_thresh = torch.quantile(torch.sigmoid(self.opacities), self.config.cull_opacity_value)
            cull_mask = torch.sigmoid(self.opacities) < cull_thresh
        elif self.config.cull_opacity_type == "absolute":
            cull_mask = torch.sigmoid(self.opacities) < self.config.cull_opacity_value
        
        cull_mask = cull_mask.squeeze()
        
        self.cull_gaussians(optimizers, cull_mask)


    def duplicate_all_existing_gaussians(self, optimizers):
        # duplicate the gaussians
        num_gaussians = self.means.shape[0]
        # mask should be of size (num_gaussians,)
        dup_mask = torch.ones(num_gaussians, dtype=torch.bool)
        self.dup_gaussians(optimizers, dup_mask)

    def cull_wayward(self, 
                     optimizers,
                     vis_before_culling : bool = False, 
                     vis_after_culling : bool = False):
        
        num_neighbors = self.config.cull_wayward_num_neighbors
        distances, indices = self.k_nearest_sklearn(self.means.data, num_neighbors)
        inds = torch.from_numpy(indices).long()
        
        dirs_to_neighbors = self.means[:, None, :] - self.means[inds]
        dirs_to_neighbors = dirs_to_neighbors / torch.norm(dirs_to_neighbors, dim=-1, keepdim=True)
        
        if self.config.cull_wayward_method == 'pca_ratio':
            
            U, S, V = torch.pca_lowrank(dirs_to_neighbors, q = 3)
            cns = S[:, 2] / S[:, 1] # If this is low that means the variance is low in the direction of the third principal component and these are the points we need
            _, sorted_inds = torch.sort(cns, descending=False)
            cull_percentile = self.config.cull_wayward_threshold_value
            num_points_to_remove = cull_percentile * len(sorted_inds)
            wayward_cull_mask = torch.zeros_like(cns, dtype = torch.bool)
            wayward_cull_mask[sorted_inds[:num_points_to_remove]] = True
            vis_colors = torch.stack([cns, cns, cns], dim=-1).detach().cpu().numpy()

        else:
            if self.config.cull_wayward_method == 'mean_distance':
                
                dists = np.mean(distances, axis=-1)
                dists_normalized = (dists - dists.min()) / (dists.max() - dists.min())

            elif self.config.cull_wayward_method == 'max_distance':
                dists = np.max(distances, axis=-1)
                dists_normalized = (dists - dists.min()) / (dists.max() - dists.min())

            if self.config.cull_wayward_threshold_type == "percentile_top":

                cull_percentile = 1 - self.config.cull_wayward_threshold_value
                cull_beyond_dist = torch.quantile(torch.from_numpy(dists), cull_percentile, interpolation='lower').item()
                wayward_cull_mask = torch.zeros_like(torch.from_numpy(dists), dtype = torch.bool)
                wayward_cull_mask[dists > cull_beyond_dist] = True

            elif self.config.cull_wayward_threshold_type == "absolute":
                cull_thresh = self.config.cull_wayward_threshold_value
                wayward_cull_mask = torch.from_numpy(dists) > cull_thresh
            
            vis_colors = np.hstack([dists_normalized[:, None], dists_normalized[:, None], dists_normalized[:, None]])

    def duplicate_high_pos_gradients(self, optimizers):
        
        grads = self.absgrads / self.absgrads_normalize_factor
        absgrads_median = torch.median(grads)
        absgrads_mean = torch.mean(grads)
        absgrads_std = torch.std(grads)
        absgrads_80percentile = torch.quantile(grads, 0.8, interpolation='lower')
        absgrads_90percentile = torch.quantile(grads, 0.9, interpolation='lower')

        grads_n = (grads - grads.min()) / (grads.max() - grads.min())
        grads_n = grads_n.detach().cpu().numpy()

        if self.config.dup_threshold_type == "percentile_top":
            duplicate_top_percentile = self.config.dup_threshold_value
            num_quantiles = int(1 / duplicate_top_percentile)
            quantiles = torch.zeros(num_quantiles)
            for i in range(1, num_quantiles):
                quantiles[i] = torch.quantile(grads, i/num_quantiles, interpolation='lower')
            
            thresh = quantiles[-1]
            # duplicate the points with top percentile
            dup_mask = grads_n > thresh
            
        elif self.config.dup_threshold_type == "absolute":
            thresh = self.config.dup_threshold_value
            dup_mask = torch.from_numpy(grads_n > thresh)
        
        self.dup_gaussians(optimizers, dup_mask)
        self.reset_absgrads()
    
    def cull_gaussians_not_projecting(self, optimizers, min_projecting_fraction = 0.1):
        
        num_gs = self.means.shape[0]
        num_frames = len(self.viewcams)

        gs_visib_matrix = torch.zeros(num_gs, num_frames, dtype=torch.bool)
        
        for idx, viewcam in enumerate(self.viewcams):

            P = viewcam.K.cpu() @ viewcam.viewmat.cpu()[:3, :4]
            w, h = viewcam.width, viewcam.height
            gaussian_means_h = torch.cat([self.means.detach().cpu(), torch.ones(num_gs, 1)], dim=-1)
            projected_means = torch.matmul(P, gaussian_means_h.t()).t()
            projected_means = projected_means[:, :2] / projected_means[:, 2:]
            projected_means_r = projected_means.round().long()
            good_inds = (projected_means_r[:, 0] >= 0) & (projected_means_r[:, 0] < w) & (projected_means_r[:, 1] >= 0) & (projected_means_r[:, 1] < h)
            projecting_within = projected_means_r[good_inds]
            projecting_on_edge = self.edge_masks[idx][projecting_within[:, 1], projecting_within[:, 0]]
            gs_visib_matrix[good_inds, idx] = projecting_on_edge
        
        mean_projections = torch.mean(gs_visib_matrix.float(), dim=1)
        cull_mask = mean_projections < min_projecting_fraction
        self.cull_gaussians(optimizers, cull_mask)

    def reset_absgrads(self):
        self.absgrads = torch.zeros(self.means.shape[0]).to(self.device)
        self.absgrads_normalize_factor = 1
    
    def update_absgrads(self):
        if self.absgrads.shape[0] != self.means.shape[0]:
            self.absgrads = torch.zeros(self.means.shape[0]).to(self.device)
            self.absgrads_normalize_factor = 1

        self.absgrads += self.xys.absgrad[0].norm(dim=-1)
        self.absgrads_normalize_factor += 1

    # Forward and load state dict

    def forward(self, idx):

        camera = self.viewcams[idx]
        outputs = self.get_outputs(camera)
        self.step += 1

        return outputs
    
    def load_state_dict(self, state_dict):
        self.gauss_params = torch.nn.ParameterDict(
            {
                "stroke_means": torch.nn.Parameter(state_dict["gauss_params.stroke_means"]),
                "stroke_scales": torch.nn.Parameter(state_dict["gauss_params.stroke_scales"]),
                "stroke_opacities": torch.nn.Parameter(state_dict["gauss_params.stroke_opacities"]),
            }
        )
    
    def export_as_ply(self, ply_path):

        ### save sampled strock points
        self.sample_gs_from_strokes()

        scales = torch.exp(self.scales).detach().cpu().numpy()
        opacities = torch.sigmoid(self.opacities).detach().cpu().numpy().reshape(-1, 1)
        means = self.means.detach().cpu().numpy()
        quats = self.quats.detach().cpu().numpy()

        cmap = torch.randint(64, 255, (self.samp_stroke_idx.max()+1, 3))
        rgb = cmap[self.samp_stroke_idx, :]

        write_gaussian_params_as_ply(means, scales, quats, opacities, ply_path, rgb=rgb)


    def export_as_json(self, json_path):

        parametric_edges_dict = {"curves_ctl_pts" : [], "lines_end_pts" : []}

        curve_means = self.gauss_params["stroke_means"][self.ptsidx_curve(), :].reshape(self.num_curves, 4, 3)      # (N1, 4, 3) Bezier curve
        line_means  = self.gauss_params["stroke_means"][self.ptsidx_line(),  :].reshape(self.num_lines, 2, 3)       # (N1, 2, 3) Line curve

        for i in range(self.num_curves):
            one_stroke = curve_means[i, :, :].detach().cpu().numpy()
            ctl_pts = [one_stroke[j, :].tolist() for j in range(4)]
            parametric_edges_dict["curves_ctl_pts"].append(ctl_pts)

        for i in range(self.num_lines):
            one_stroke = line_means[i, :, :].detach().cpu().numpy()
            ctl_pts = one_stroke[0, :].tolist() + one_stroke[1, :].tolist()
            parametric_edges_dict["lines_end_pts"].append(ctl_pts)

        with open(json_path, "w") as f:
            json.dump(parametric_edges_dict, f)
