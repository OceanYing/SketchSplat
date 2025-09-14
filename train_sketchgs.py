import os
import torch
import random
import time
import argparse
import ipdb
import numpy as np
import cv2
from shutil import copyfile

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from sketchgaussians.data.dataset import InputDataset
from sketchgaussians.utils import parse_utils, train_utils, data_utils
from sketchgaussians.vis import vis_utils
from sketchgaussians.models.sketch_gs import SketchSplatting
from sketchgaussians.data.dataparsers import DataParserFactory

def train_epoch(model: SketchSplatting, 
                dataloader, 
                optimizers,
                device,
                summary_writer,
                epoch,
                num_epochs,
                projection_loss_config,
                orientation_loss_config,
                weights_update_freq = 1,
                output_dir=None,
                topo_config=None,
            ):

    lambda_projection_loss = lambda_dir_loss = lambda_ratio_loss = 1.0
    bg_edge_pixel_ratio = train_utils.get_bg_edge_pixel_ratio(loss_config=projection_loss_config, 
                                                  step = epoch, 
                                                  max_steps = num_epochs)
    lambda_projection_loss = train_utils.get_lambda_projection(loss_config=projection_loss_config,
                                                   step = epoch,
                                                   max_steps = num_epochs)

    direction_loss_start_at = orientation_loss_config["start_dir_loss_at_epoch"]
    ratio_loss_start_at = orientation_loss_config["start_ratio_loss_at_epoch"]
    ratio_loss_scale_factor = orientation_loss_config["ratio_loss_scale_factor"]
    direction_loss_scale_factor = orientation_loss_config["dir_loss_scale_factor"]
    apply_ratio_loss = False
    apply_direction_loss = False

    div = 1
    if epoch > direction_loss_start_at:
        div += direction_loss_scale_factor
    if epoch > ratio_loss_start_at:
        div += ratio_loss_scale_factor
    else:
        div = 1
    
    for _,opt in optimizers.items():
        opt.zero_grad()

    avg_loss = 0

    sampling_whole_num_epochs_ratio = projection_loss_config["sampling_whole_num_epochs_ratio"]
    pixel_sampling = projection_loss_config["loss_before_alternating"]

    if epoch > projection_loss_config["start_alternating_at_epoch"]:
        check_pixel_sampling = True
    else:
        check_pixel_sampling = False

    if 'topo_concentration' in topo_config.keys():
        dir_loss_flag = topo_config['topo_concentration']
    else:
        dir_loss_flag = False
    
    if (epoch > direction_loss_start_at) and dir_loss_flag:
        apply_direction_loss = True
    
    if epoch > ratio_loss_start_at:
        apply_ratio_loss = True
    
    projection_loss_all = 0

    model.sample_gs_from_strokes()  # for each epoch, sample points from strokes for one time only

    for i, data in enumerate(dataloader):

        if check_pixel_sampling:
            if model.step % sampling_whole_num_epochs_ratio == 0:
                pixel_sampling = projection_loss_config['less_freq_loss']
            else:
                pixel_sampling = projection_loss_config['more_freq_loss']
        
        # get rendered image
        idx = data['idx']
        output = model(idx)

        output_image = output['rgb']
        output_image = output_image[:,:,0].unsqueeze(0)

        # get ground truth image
        gt_image = data['image']/255.0

        gt_image = gt_image.to(device)

        # compute projection loss
        projection_loss = model.compute_projection_loss(output_image[0,:,:], gt_image[0,:,:], 
                                                        image_index=idx,
                                                        strategy=pixel_sampling,
                                                        bg_edge_pixel_ratio = bg_edge_pixel_ratio)
        
        summary_writer.add_scalar('Projection loss', projection_loss.item(), epoch)

        projection_loss_ = lambda_projection_loss * projection_loss
        avg_loss += projection_loss.item()
        
        projection_loss_all += projection_loss / len(dataloader)

    projection_loss_all.backward()

    for param,opt in optimizers.items():
        opt.step()
        opt.zero_grad()                

    avg_loss /= len(dataloader)

    if (output_dir != None) and (epoch % 5 == 0):
        log_render_path = os.path.join(output_dir, "log_render")
        os.makedirs(log_render_path, exist_ok=True)

        save_img = torch.cat([gt_image[0], output_image[0]], dim=1).detach().cpu().numpy()
        save_img = (np.clip(save_img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(log_render_path, '{:0>4d}_{:0>2d}.png'.format(epoch, int(idx))), save_img)
        
    return avg_loss


def train(model: SketchSplatting, config, dataloader, log_dir, output_dir, device):
    summary_writer = SummaryWriter(log_dir)
    
    optim_config = config["optim"]
    loss_config = config["loss"]
    num_epochs = config["num_epochs"]
    weights_update_freq = config["weights_update_freq"]
    topo_config = config["topo"]

    optimizers, schedulers = train_utils.get_optimizers_schedulers(model = model,
                                                       config = optim_config)
    print("Optimizers and schedulers created")

    projection_loss_config = loss_config["projection_losses"]
    orientation_loss_config = loss_config["orientation_losses"]

    model.dir_loss_num_nn = orientation_loss_config["dir_loss_num_nn"]
    model.dir_loss_enforce_method = orientation_loss_config["dir_loss_enforce_method"]
    
    with tqdm(total=num_epochs, desc=f"Training", unit='epoch') as pbar:
        for epoch in range(num_epochs):

            if epoch == 0:
                if topo_config['topo_connect_endpoints']:
                    model.topo_connect_endpoints(optimizers, threshold=0.01)
                if topo_config['topo_merge_colinear_sketch']:
                    model.topo_merge_colinear_sketch(optimizers, angle_thres=5, offset_tol=0.01, overlap_epsilon=0.01)
            
            avg_loss = train_epoch(model, 
                                dataloader, 
                                optimizers, 
                                device,
                                summary_writer,
                                epoch, 
                                num_epochs,
                                projection_loss_config,
                                orientation_loss_config,
                                weights_update_freq,
                                output_dir,
                                topo_config)
            
            pbar.set_postfix({'Loss': avg_loss, 
                              "Num Gaussians": model.gauss_params["stroke_means"].shape[0]})
            pbar.update(1)
            update_nn = False
            reset_absgrads = False

            for _,sch in schedulers.items():
                sch.step()

            if (epoch > 0) and (epoch % 50 == 0):
                
                if topo_config['topo_connect_endpoints']:
                    connect_dist_thres = 0.01 if ('connect_dist_thres' not in topo_config.keys()) else topo_config['connect_dist_thres']
                    model.topo_connect_endpoints(optimizers, 
                                                 threshold=connect_dist_thres)

                if topo_config['topo_merge_overlapping_sketch']:
                    overlapping_dist_thres = 0.02 if ('overlapping_dist_thres' not in topo_config.keys()) else topo_config['overlapping_dist_thres']
                    overlapping_ratio_thres = 0.85 if ('overlapping_ratio_thres' not in topo_config.keys()) else topo_config['overlapping_ratio_thres']
                    model.topo_merge_overlapping_sketch(optimizers, 
                                                        threshold=overlapping_dist_thres, 
                                                        overlap_ratio=overlapping_ratio_thres)

                if topo_config['topo_merge_colinear_sketch']:
                    colinear_angle_thres = 5 if ('colinear_angle_thres' not in topo_config.keys()) else topo_config['colinear_angle_thres']
                    colinear_offset_thres = 0.01 if ('colinear_offset_thres' not in topo_config.keys()) else topo_config['colinear_offset_thres']
                    colinear_overlap_thres = 0.01 if ('colinear_overlap_thres' not in topo_config.keys()) else topo_config['colinear_overlap_thres']
                    model.topo_merge_colinear_sketch(optimizers, 
                                                     angle_thres=colinear_angle_thres, 
                                                     offset_tol=colinear_offset_thres, 
                                                     overlap_epsilon=colinear_overlap_thres)

            if (epoch > 199) and (epoch % 100 == 0):
                if 'topo_transfer' in topo_config.keys() and topo_config['topo_transfer']:
                    model.topo_transfer_curve_to_line(optimizers, threshold=0.95)

            if (epoch > 199) and (epoch % 50 == 0):
                if topo_config['topo_filter_low_opacity']:
                    model.topo_filter_low_opacity(optimizers)

            if (epoch % 100 == 0) and (epoch < 500):
                if topo_config['topo_add_new_sketch']:
                    model.topo_add_new_sketch(optimizers, new_length=0.02)

            if epoch == (num_epochs - 1):
                if topo_config['topo_filter_low_opacity']:
                    model.topo_filter_low_opacity(optimizers)
                if topo_config['topo_filter_not_projecting']:
                    model.topo_filter_not_projecting(optimizers, min_projecting_fraction=0.1)


            log_ply_path = os.path.join(output_dir, "log_ply")
            os.makedirs(log_ply_path, exist_ok=True)

            if (epoch % 50 == 0):
                model.export_as_ply(os.path.join(log_ply_path, "debug_epoch{:0>3d}.ply".format(epoch)))
    
    train_utils.save_model(model, output_dir, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the config file")
    parser.add_argument("--ckpt_path", type=str, help="Load from pretrained checkpoint at this path", default=None)
    parser.add_argument("--scene_name", type=str, help="Name of the experiment", default=None)
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun the training", default=False)
    parser.add_argument("--sub_exp_name", type=str, help="Name of the experiment", default=None)
    parser.add_argument("--stroke_file_path", type=str, help="stroke_file_path (.json)", default=None)
    parser.add_argument("--loadrgb", action="store_true", help="load RGB images", default=False)
    parser.add_argument("--sketch_add_noise", type=float, help="sketch_add_noise (std of Gaussian White Noise)", default=0.0)
    
    args = parser.parse_args()

    ### set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Data Parse
    config_file = args.config_file
    scene_name = args.scene_name
    force_rerun = True if args.force_rerun else False
    stroke_file_path = args.stroke_file_path
    sketch_add_noise = args.sketch_add_noise

    # Load config
    model_config, training_config, data_config, output_config = parse_utils.get_configs(config_file)
    model_config["stroke_json_path"] = stroke_file_path
    model_config["sketch_add_noise"] = sketch_add_noise
    
    # get data parser
    dataparser, images_dir, seed_points_path, rgb_dir = parse_utils.parse_data(data_config, scene_name)

    # init seed points
    if not model_config["init_random_init"]:
        seed_points = data_utils.init_seed_points_from_file(model_config, seed_points_path)
    else:
        num_seed_points = model_config["init_min_num_gaussians"]
        if "random_init_box_center" in model_config:
            box_center = model_config["random_init_box_center"]
        else:
            box_center = 0.5

        box_size = model_config["random_init_box_size"]
        num_seed_points = model_config["init_min_num_gaussians"]
        seed_points = data_utils.init_seed_points_random(num_seed_points, box_center, box_size)

    # initialize views and get scale factor if needed
    parser_type = data_config["parser_type"]
    image_res_scaling_factor = data_config["image_res_scaling_factor"]

    data_utils.init_views(dataparser, 
                             images_dir, 
                             rgb_dir if (args.loadrgb is True) else None,
                             parser_type = parser_type,
                             image_res_scaling_factor = image_res_scaling_factor)

    # Scale and translate seed points
    if (data_config["scale_scene_unit"]):
        # get scale from cameras
        rotmats = [view['camera'].R for view in dataparser.views]
        tvecs = [view['camera'].t for view in dataparser.views]
        scale = data_utils.get_scale_from_cameras(rotmats, tvecs)
        # if seed points exist, get scale from seed points
        
        if seed_points is not None:
            points_scale = data_utils.get_scale_from_points(seed_points, min_percentile=0.05, max_percentile=0.95)
            # use the maxiumum to scale both seed points and views
            scale = max(scale, points_scale)

        # scale points and cameras
        seed_points = seed_points * 1/scale
        for view in dataparser.views:
            view['camera'].scale_translation(1/scale)
    
    # Set up the model and the dataloader and appropriate paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    views = [view['camera'] for view in dataparser.views]
    gt_images = [view['image']/255.0 for view in dataparser.views]

    for view in views:
        view.to(device)

    model = SketchSplatting()
    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path))
    else:
        model.poplutate_params(seed_points=seed_points, viewcams=views, config=model_config)
    
    print("Model populated")
    
    model.compute_image_masks(gt_images)
    model.compute_weight_masks()
    model.load_sketches()
    model.to(device)

    # Create the dataloader
    dataset = InputDataset(dataparser=dataparser)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Loaded {len(dataloader)} samples in the dataloader")

    output_base = output_config["output_dir"]

    edge_detector = data_config["edge_detection_method"]
    exp_name = output_config["exp_name"] + "_" + edge_detector
    if args.sub_exp_name != None:
        output_dir = os.path.join(output_base, exp_name, scene_name+"_"+args.sub_exp_name)
        log_dir = os.path.join(output_config["log_dir"], exp_name, scene_name+"_"+args.sub_exp_name)
    else:
        output_dir = os.path.join(output_base, exp_name, scene_name)
        log_dir = os.path.join(output_config["log_dir"], exp_name, scene_name)
    start_time = time.time()

    file_backup(output_dir)


    ### save the input sketches (for adding noise, store the one with added noisy)
    if output_config["export_ply"]:
        output_ply_path = os.path.join(output_dir, "gaussians_input.ply")
        model.export_as_ply(output_ply_path)

        output_json_path = os.path.join(output_dir, "strokes_input.json")
        model.export_as_json(output_json_path)


    # Train the model
    num_epochs = training_config["num_epochs"]
    max_epochs_weights_file = os.path.join(output_dir, f"{exp_name}_epoch{num_epochs-1}.pth")
    if os.path.exists(max_epochs_weights_file):
        if not force_rerun:
            print(f"Model already trained for {num_epochs} epochs. Exiting")
            return 0
    
    train(model=model,
          config = training_config,
          dataloader=dataloader, 
          log_dir=log_dir,
          output_dir=output_dir,
          device=device)

    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds")
    with open(os.path.join(output_dir, "time.txt"), "w") as f:
        f.write(f"Training took {end_time - start_time} seconds")

    if output_config["export_ply"]:
        output_ply_path = os.path.join(output_dir, "gaussians_all.ply")
        model.export_as_ply(output_ply_path)

        output_json_path = os.path.join(output_dir, "strokes_all.json")
        model.export_as_json(output_json_path)


def file_backup(base_exp_dir):
    # copy python file
    dir_lis = [
      "configs",
      "edgegaussians",
      "sketchgaussians",
      "scripts",
      "eval_edgegs.py",
      "eval_sketchgs.py",
      "fit_edges.py",
      "run_*.sh"
    ]
    cur_dir = os.path.join(base_exp_dir, "recording")
    os.makedirs(cur_dir, exist_ok=True)
    files = os.listdir("./")
    for f_name in files:
        if f_name[-3:] == ".py":
            copyfile(os.path.join("./", f_name), os.path.join(cur_dir, f_name))

    for dir_name in dir_lis:
        os.system(f"cp -r {dir_name} {cur_dir}")


if __name__ == "__main__":
    main()