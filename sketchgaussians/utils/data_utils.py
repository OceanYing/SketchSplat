import numpy as np
import torch
import open3d as o3d
import ipdb

from sketchgaussians.utils.colmap_read_write_model import read_points3D_text, read_points3d_binary

from scipy.spatial.distance import cdist


def init_views(dataparser, 
                images_dir, 
                rgb_dir = None, 
                parser_type:str = 'emap',
                image_res_scaling_factor:float = 1.0):
    
    if parser_type == "colmap":
        if image_res_scaling_factor is not None:
            image_res_scaling_factor = image_res_scaling_factor
        else:
            image_res_scaling_factor = 1.0
        print(f"Scaling the images by a factor of {image_res_scaling_factor}")
        scene_scale = dataparser.load_views(images_dir=images_dir, rgb_dir=rgb_dir,
                                            image_res_scaling_factor=image_res_scaling_factor)
        print(f"Loaded {len(dataparser.views)} views")
    else:
        scene_scale = dataparser.load_views(images_dir=images_dir, rgb_dir=rgb_dir)
    
    return scene_scale

def init_seed_points_from_file(model_config, seed_points_path,
                     box_center: float = 0.5,
                     box_size: float = 1.0):
    
    if seed_points_path.endswith(".txt"):
        try:
            seed_points = torch.tensor(np.loadtxt(seed_points_path)).float()
        except:
            points3d = read_points3D_text(seed_points_path)
            sparse_pc = np.array([points3d[point].xyz for point in points3d])
            seed_points = torch.tensor(sparse_pc).float()

    elif seed_points_path.endswith(".ply"):
        sparse_pc = o3d.io.read_point_cloud(seed_points_path)
        points = np.asarray(sparse_pc.points).reshape(-1,3)    
        seed_points = torch.tensor(points).float()

    elif seed_points_path.endswith(".bin"):
        points3d = read_points3d_binary(seed_points_path)
        sparse_pc = np.array([points3d[point].xyz for point in points3d])
        seed_points = torch.tensor(sparse_pc).float()
    
    num_seed_points = seed_points.shape[0]
    if num_seed_points < model_config["init_min_num_gaussians"]:
        # Currently only implemented if we assume that the scene is centered at 0.5, 0.5, 0.5 and has a size of 1.0
        # mean_cam_center = torch.mean(torch.stack([dataparser.get_view_center(view) for view in dataparser.views]), dim=0).reshape(-1,3)
        
        num_sample_more = model_config["init_min_num_gaussians"] - num_seed_points
        
        # randomly sample extra points - works well if you know the extent of the scene
        # seed_points_center = torch.median(seed_points) 
        # seed_points_extra =  box_size*(torch.rand((num_sample_more, 3)).float()) -box_size/2 + seed_points_center
        
        replication_factor = int(np.ceil(num_sample_more/num_seed_points))
        noise = 0.1*torch.randn((replication_factor * num_seed_points, 3)).float()
        seed_points_extra = torch.cat([seed_points]*replication_factor, dim=0) + noise
        seed_points = torch.cat([seed_points, seed_points_extra], dim=0)
            
    print(f"Loaded {seed_points.shape[0]} seed points")

    return seed_points

def init_seed_points_random(num_points, box_center, box_size):
    
    seed_points =  box_size*(torch.rand((num_points, 3)).float()) - box_size/2 + box_center
    return seed_points

def scale_seed_points(seed_points, scene_scale):
    print("Scaling the poionts with factor ", scene_scale)
    # print("Translation the points with factor ", scene_translation)
    seed_points = scene_scale * (seed_points) # + scene_translation)

    return seed_points

def get_scale_from_cameras(rotmats, transvecs):
    
    '''
    rotmats : list pf 3x3 numpy matrices such that determinant of each matrix is +1
    transvecs : list of 3x1 numpy matrices
    '''
    cam_centers = []
    num_cams = len(rotmats)
    assert num_cams == len(transvecs)
    for i in range(num_cams):
        cam_centers.append(-rotmats[i].T @ transvecs[i])

    cam_centers_array = np.array(cam_centers)   

    distances_between_cams = cdist(cam_centers_array, cam_centers_array)
    max_distance = np.max(distances_between_cams)
    
    cameras_scale = max_distance
    
    return cameras_scale

def get_scale_from_points(points, min_percentile, max_percentile):
    
    points_extent = torch.quantile(points, max_percentile, dim=0) - torch.quantile(points, min_percentile, dim=0)
    points_scale = torch.max(points_extent)
    
    return points_scale



import torch
import torch.nn.functional as F

def soft_dilate(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Applies a non-strict dilation operation to a float image using max pooling.
    
    Args:
        image (torch.Tensor): Input float image of shape (B, C, H, W).
        kernel_size (int): Size of the dilation kernel (default: 3x3).
        
    Returns:
        torch.Tensor: Dilated image with the same shape as input.
    """
    # Ensure the input is float
    if not image.is_floating_point():
        image = image.float()

    # Apply max pooling with stride=1 and appropriate padding to retain shape
    padding = kernel_size // 2  # Keep same spatial size
    dilated = F.max_pool2d(image, kernel_size, stride=1, padding=padding)

    return dilated


def soft_dilate_with_mask(image: torch.Tensor, mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Applies a non-strict dilation operation to a float image, restricted to masked regions.
    
    Args:
        image (torch.Tensor): Input float image of shape (B, C, H, W).
        mask (torch.Tensor): Binary mask (same shape as image), where 1 indicates areas to process.
        kernel_size (int): Size of the dilation kernel (default: 3x3).
        
    Returns:
        torch.Tensor: Image with dilation applied only in masked regions.
    """
    # Ensure the input is float
    if not image.is_floating_point():
        image = image.float()

    # Apply max pooling for dilation
    padding = kernel_size // 2
    dilated = F.max_pool2d(image, kernel_size, stride=1, padding=padding)

    # Apply the mask: retain original values where mask == 0
    output = image * (1 - mask) + dilated * mask

    return output
