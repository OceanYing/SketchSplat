#!/bin/bash

folder_list=($(ls -d ./data/ABC-NEF_Edge/data/* | sort))
total_folders=${#folder_list[@]}

echo "Found $total_folders scenes."


for data_path in "${folder_list[@]}"; do
    scene_name=$(basename "$data_path")
    echo ">>> Processing scene: $scene_name"


    ### 1. Training EdgeGS
    config_file_path=./configs/edgegs/ABC_2DGSstb.json
    python train_edgegs.py --config_file "$config_file_path" --scene_name "$scene_name" --sub_exp_name edgegs
    python fit_edges.py --config_file "$config_file_path" --scene_name "$scene_name" --sub_exp_name edgegs --save_filtered --save_sampled_points


    ### 2. Training SketchSplat
    echo "Processing SketchGS-Recon: $scene_name"
    config_file_path=./configs/sketchsplat/ABC_2DGSstb_sketchsplat.json
    stroke_init_path=./output/ABC/edgegs_2DGSstb/${scene_name}_edgegs/fitting/parametric_edges.json
    python train_sketchgs.py --config_file "$config_file_path" --scene_name "$scene_name" --sub_exp_name sketchgs --stroke_file_path "$stroke_init_path"


### 3. To evaluate EdgeGS
python eval_edgegs.py --dataset ABC --scan_names all --version edgegs --use_parametric_edges --sub_exp_name edgegs --edge_detector 2DGSstb --gt_base_dir /fs/vulcan-projects/SER/SketchSplat/SketchSplatting_surface/data/ABC-NEF_Edge/groundtruth


### 4. To evaluate SketchSplat
python eval_sketchgs.py --dataset ABC --scan_names all --version sketchgs --sub_exp_name sketchgs --edge_detector 2DGSstb --gt_base_dir ./data/ABC-NEF_Edge/groundtruth
