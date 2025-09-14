import os
import json
import numpy as np
import open3d as o3d
import cv2
import ipdb
import glob
from PIL import Image
import torch
import torch.nn.functional as F

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def load_from_json(filename):
    """Load a dictionary from a JSON filename."""
    assert filename.split(".")[-1] == "json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def project2D_single(K, R, T, points3d):
    shape = points3d.shape
    assert shape[-1] == 3
    X = points3d.reshape(-1, 3)

    x = K @ (R @ X.T + T)
    x = x.T
    x = x / x[:, -1:]
    x = x.reshape(*shape)[..., :2].reshape(-1, 2).tolist()
    return x

def project3D_single(K, R, T, points3d):
    shape = points3d.shape
    assert shape[-1] == 3
    X = points3d.reshape(-1, 3)

    x = K @ (R @ X.T + T)
    x = x.T
    uv = x / x[:, -1:]
    uv = uv.reshape(*shape)[..., :2].reshape(-1, 2).tolist()
    z = x.reshape(*shape)[..., -1].reshape(-1, 1)
    return uv, z

def project2D(K, R, T, all_curve_points, all_line_points):
    all_curve_uv, all_line_uv = [], []
    for curve_points in all_curve_points:
        curve_points = np.array(curve_points).reshape(-1, 3)
        curve_uv = project2D_single(K, R, T, curve_points)
        all_curve_uv.append(curve_uv)
    for line_points in all_line_points:
        line_points = np.array(line_points).reshape(-1, 3)
        line_uv = project2D_single(K, R, T, line_points)
        all_line_uv.append(line_uv)
    return all_curve_uv, all_line_uv

def load_images_and_cameras(dataparser, rgb_flag=False):

    views = [view['camera'] for view in dataparser.views]
    edge_images = [view['image']/255.0 for view in dataparser.views]
    cameras = [None for _ in range(len(views))]
    h, w = views[0].height, views[0].width

    for i in range(len(views)):
        K = views[i].get_K().cpu().numpy()
        viewmat = views[i].viewmat.cpu().numpy()
        R = viewmat[:3, :3]
        t = viewmat[:3, 3:]
        cameras[i] = {'K' : K, 'R' : R, 't' : t, 'h' : h, 'w' : w}

    if rgb_flag:
        rgba_images = [view['rgb'] for view in dataparser.views]
        return edge_images, cameras, rgba_images
    else:
        return edge_images, cameras


def filter_stat_outliers(means : np.ndarray, num_nn : int = 10, std_multiplier: float = 3.0):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    print("Statistical oulier removal")
    cl, inlier_inds = pcd.remove_statistical_outlier(nb_neighbors=num_nn,
                                                        std_ratio=std_multiplier)
    
    print(f"Removed {len(means) - len(inlier_inds)} outliers")

    return np.array(inlier_inds).reshape(-1)

def filter_by_opacity(opacities : np.ndarray, min_opacity : float):
    num_pts = opacities.shape[0]
    print(f"Num points before filtering by opacity: {num_pts}")
    inlier_inds =  opacities > min_opacity
    print(f"Removed {len(opacities) - np.sum(inlier_inds)} points")

    return inlier_inds.reshape(-1)


def filter_by_projection(gaussian_means,
                         edge_images,
                         cameras,
                         visib_thresh:float = 0.1):

    num_gs = gaussian_means.shape[0]
    num_images = len(edge_images)
    print(f"Num points before filtering by projection: {num_gs}")
    gs_visib_matrix = np.zeros((num_gs, num_images))

    for i in range(num_images):
        
        K = cameras[i]['K']
        R = cameras[i]['R']
        t = cameras[i]['t']
        h = cameras[i]['h']
        w = cameras[i]['w']
        all_curve_uv, _ = project2D(
            K, R, t, [gaussian_means], []
        )
        edge_uv = all_curve_uv[0]
        edge_uv = np.array(edge_uv)
        if len(edge_uv) == 0:
            continue
        edge_uv = np.round(edge_uv).astype(np.int32)
        edge_u = edge_uv[:, 0]
        edge_v = edge_uv[:, 1]

        edge_map = edge_images[i].cpu().numpy()

        valid_mask = (edge_u >= 0) & (edge_u < w) & (edge_v >= 0) & (edge_v < h)
        valid_edge_uv = edge_uv[valid_mask,:]

        if len(valid_edge_uv) > 0:
            
            projected_edge = edge_map[valid_edge_uv[:, 1], valid_edge_uv[:, 0]]
            gs_visib_matrix[valid_mask, i] += projected_edge

    gs_visib = np.mean(gs_visib_matrix, axis=1)
    inlier_inds = gs_visib > visib_thresh

    print(f"Removed {num_gs - np.sum(inlier_inds)} points")

    return inlier_inds.reshape(-1)


def filter_by_projection_alpha(gaussian_means,
                               edge_images,
                               rgba_images,
                               cameras,
                               visib_thresh:float = 0.1,
                               inside_thres:float = 0.1):

    num_gs = gaussian_means.shape[0]
    num_images = len(edge_images)
    print(f"Num points before filtering by projection: {num_gs}")
    gs_visib_matrix = np.zeros((num_gs, num_images))
    gs_outside_matrix = np.zeros((num_gs, num_images))

    for i in range(num_images):
        
        K = cameras[i]['K']
        R = cameras[i]['R']
        t = cameras[i]['t']
        h = cameras[i]['h']
        w = cameras[i]['w']
        all_curve_uv, _ = project2D(
            K, R, t, [gaussian_means], []
        )
        edge_uv = all_curve_uv[0]
        edge_uv = np.array(edge_uv)
        if len(edge_uv) == 0:
            continue
        edge_uv = np.round(edge_uv).astype(np.int32)
        edge_u = edge_uv[:, 0]
        edge_v = edge_uv[:, 1]

        edge_map = edge_images[i].cpu().numpy()
        rgba_map = rgba_images[i].cpu().numpy()
        alpha_map = rgba_map[..., 3]

        valid_mask = (edge_u >= 0) & (edge_u < w) & (edge_v >= 0) & (edge_v < h)
        valid_edge_uv = edge_uv[valid_mask,:]

        if len(valid_edge_uv) > 0:
            
            projected_edge = edge_map[valid_edge_uv[:, 1], valid_edge_uv[:, 0]]
            gs_visib_matrix[valid_mask, i] += projected_edge

            projected_edge = alpha_map[valid_edge_uv[:, 1], valid_edge_uv[:, 0]]
            gs_outside_matrix[valid_mask, i] += (projected_edge < inside_thres)*1.0


    gs_visib = np.mean(gs_visib_matrix, axis=1)
    inlier_inds = (gs_visib > visib_thresh) * (gs_outside_matrix.sum(axis=1) < num_images * 0.1)

    print(f"{num_gs - np.sum(inlier_inds)} points should be removed")

    return inlier_inds.reshape(-1)


def filter_by_projection_dep(gaussian_means,
                        edge_images,
                        cameras,
                        visib_thresh:float = 0.1, scene_name:str=None):

    num_gs = gaussian_means.shape[0]
    num_images = len(edge_images)
    print(f"Num points before filtering by projection: {num_gs}")
    gs_visib_matrix = np.zeros((num_gs, num_images))
    gs_visib_matrix_dep = np.zeros((num_gs, num_images))

    deps, depvars_m = load_dep_varm(scene_name)

    for i in range(num_images):
        
        K = cameras[i]['K']
        R = cameras[i]['R']
        t = cameras[i]['t']
        h = cameras[i]['h']
        w = cameras[i]['w']

        # all_curve_uv, _ = filtering.project2D(
        #     K, R, t, [gaussian_means], []
        # )
        # edge_uv = all_curve_uv[0]

        edge_uv, z = project3D_single(
            K, R, t, gaussian_means.numpy()
        )

        edge_uv = np.array(edge_uv)
        if len(edge_uv) == 0:
            continue
        edge_uv = np.round(edge_uv).astype(np.int32)
        edge_u = edge_uv[:, 0]
        edge_v = edge_uv[:, 1]

        edge_map = edge_images[i].cpu().numpy()

        all_idx = np.arange(num_gs)

        valid_mask = (edge_u >= 0) & (edge_u < w) & (edge_v >= 0) & (edge_v < h)
        valid_edge_uv = edge_uv[valid_mask,:]
        all_idx = all_idx[valid_mask]

        dep_samps = deps[i, valid_edge_uv[:, 1], valid_edge_uv[:, 0]]
        d_mask = (np.abs(z[...,0] - dep_samps) < 0.02)
        valid_mask[all_idx[~d_mask]] = False
        valid_edge_uv = valid_edge_uv[d_mask, :]
        # print(all_idx.shape, valid_mask.shape, valid_mask.sum(), valid_edge_uv.shape)

        valid_idx = np.arange(num_gs)[valid_mask]

        if len(valid_edge_uv) > 0:
            projected_edge = edge_map[valid_edge_uv[:, 1], valid_edge_uv[:, 0]]
            gs_visib_matrix[valid_mask, i] += projected_edge


            ## Analysis: for some edges, it may happen that edge value is low under the best view (low variance)
            onedge_mask = projected_edge > 0.05
            valid_onedge_uv = valid_edge_uv[onedge_mask, :]
            # dv_mask = depvars[i] < 1e-3
            dv_mask = ~depvars_m[i] / 255.0     # depth variance low enough
            projected_dep = dv_mask[valid_onedge_uv[:, 1], valid_onedge_uv[:, 0]] * 1.0
            gs_visib_matrix_dep[valid_idx[onedge_mask], i] += projected_dep

    gs_visib = np.mean(gs_visib_matrix, axis=1)
    inlier_inds = gs_visib > visib_thresh

    print(gs_visib_matrix_dep.shape)
    gs_visib_dep = np.sum(gs_visib_matrix_dep, axis=1)
    inlier_inds = (gs_visib_dep >= 1) | inlier_inds

    print(f"Removed {num_gs - np.sum(inlier_inds)} points")

    return inlier_inds.reshape(-1)
    # return inlier_inds.reshape(-1), gs_visib_dep


def load_dep_varm(scene_name):
    ### load data
    scene_p = f'/fs/vulcan-projects/SER/SketchSplat/SketchSplatting_surface/data/ABC-NEF_Edge/data/{scene_name}'
    # scene_p = f'/fs/vulcan-projects/SER/SketchSplat/SketchSplatting_surface/data/ABC-NEF-roundbound/data/{scene_name}'
    imgs_p = [os.path.join(scene_p, f'color/{i}_colors.png') for i in range(0, 50)]

    ### load 2DGS results
    gs2d_p = f'/fs/vulcan-projects/SER/SketchSplat/SketchSplatting_surface/gs2d/output/batchrun_1/nef_250217_{scene_name}'
    # gs2d_p = f'/fs/vulcan-projects/SER/SketchSplat/SketchSplatting_surface/gs2d/output/batchrun_2/nef_250217_{scene_name}'
    deps_p = sorted(glob.glob(os.path.join(gs2d_p, 'train/ours_30000/depth_raw/*')))
    deps = []
    alps = []

    for i in range(len(deps_p)):
        depth = Image.open(deps_p[i])
        deps.append(np.array(depth))

        img = Image.open(imgs_p[i])
        alps.append(np.array(img)[..., -1])

    deps = np.stack(deps, axis=0)
    alps = np.stack(alps, axis=0)


    depvars = []
    depvars_m = []

    for i in range(len(deps_p)):
        dep_alp = deps[i] * (alps[i] / 255.0)
        depvar = compute_variance_image(torch.from_numpy(dep_alp)[None, None, :, :].float(), kernel_size=3).squeeze().numpy()
        depvars.append(depvar)

        # mask = depvar > 1e-4
        mask = depvar > 1e-3
        # mask = depvar > 1e-1
        kernel = np.ones((3, 3), np.uint8)
        dep_mask = cv2.dilate(np.uint8((mask)*255.0), kernel, iterations=3)
        depvars_m.append(dep_mask)

    depvars = np.stack(depvars, axis=0)
    depvars_m = np.stack(depvars_m, axis=0)

    return deps, depvars_m





def compute_variance_image(image, kernel_size=5):
    """
    Compute the variance image for a single-channel image using PyTorch.
    Only non-zero values within the sliding window are considered for variance calculation.
    
    Args:
        image (torch.Tensor): Input single-channel image with shape (1, 1, H, W).
        kernel_size (int): Size of the sliding window for local variance computation (default is 5).
    
    Returns:
        variance_image (torch.Tensor): Variance image with the same shape as the input, (1, 1, H, W).
    """
    # Padding size for the convolution
    padding = kernel_size // 2

    # Create a binary mask indicating non-zero values
    non_zero_mask = (image != 0).float()
    
    # Count the number of non-zero values in each window
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(non_zero_mask.device)
    non_zero_count = F.conv2d(non_zero_mask, kernel, padding=padding)
    
    # Avoid division by zero
    non_zero_count[non_zero_count == 0] = 1

    # Compute the sum of the image values within the sliding window
    sum_image = F.conv2d(image, kernel, padding=padding)
    
    # Compute the mean of non-zero values within each sliding window
    mean_image = sum_image / non_zero_count

    # Compute the sum of squared image values within the sliding window
    squared_image = image ** 2
    sum_squared_image = F.conv2d(squared_image, kernel, padding=padding)

    # Compute the mean of squared non-zero values
    mean_squared_image = sum_squared_image / non_zero_count

    # Compute the variance using the formula: variance = mean_squared - mean^2
    variance_image = mean_squared_image - mean_image ** 2

    # Set variance to zero where the window contains only zeros
    variance_image[non_zero_mask == 0] = 0

    return variance_image