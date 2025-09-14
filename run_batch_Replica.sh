#!/bin/bash

folder_list=($(ls -d ./data/Replica_Edge/* | sort))
total_folders=${#folder_list[@]}

echo "Found $total_folders scenes."


for data_path in "${folder_list[@]}"; do
    scene_name=$(basename "$data_path")
    echo ">>> Processing scene: $scene_name"


    ### 1. Training EdgeGS
    python train_edgegs.py --config_file configs/edgegs/Replica.json --scene_name $scene_name --sub_exp_name edgegs
    python fit_edges.py --config_file configs/edgegs/Replica.json --scene_name $scene_name --sub_exp_name edgegs --save_filtered --save_sampled_points


    ### 2. Training SketchSplat
    echo "Processing SketchGS-Recon: $scene_name"
    stroke_init_path=./output/Replica/edgegs_PidiNet/${scene_name}_edgegs/fitting/parametric_edges.json
    python train_sketchgs.py \
        --config_file configs/sketchsplat/Replica_sketchsplat.json \
        --scene_name $scene_name \
        --sub_exp_name sketchgs \
        --stroke_file_path $stroke_init_path
fi
