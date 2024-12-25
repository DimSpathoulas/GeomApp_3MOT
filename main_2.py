from __future__ import print_function

import argparse
import copy
import json
import pickle
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
from functions.outer_funcs import create_box_annotations, format_sample_result
from g1_g4 import TrackerNN
from nuscenes import NuScenes
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_tracker_states(Tracker, save_path):
    """
    Save states of G1, G2, G3, and G4 from Tracker to a single file.
    
    Args:
    Tracker: The Tracker object containing G1, G2, G3, and G4 models.
    save_path (str): Path to save the model states.
    """
    state_dict = {
        'G1': Tracker.G1.state_dict(),
        'G4': Tracker.G4.state_dict()
    }
    torch.save(state_dict, save_path)
    print(f"Saved G1, G4 states to {save_path}")


def load_tracker_states(Tracker, load_path):
    """
    Load states of G1 and G4 into Tracker from a single file.
    
    Args:
    Tracker: The Tracker object containing G1 and G4 models.
    load_path (str): Path to load the model states from.
    """
    if os.path.exists(load_path):
        state_dict = torch.load(load_path)
        Tracker.G1.load_state_dict(state_dict['G1'])
        Tracker.G4.load_state_dict(state_dict['G4'])
        print(f"Loaded G1, G4 states from {load_path}")
    else:
        print(f"No saved states found at {load_path}")

    return Tracker


def project_to_svd_space(features, svd_matrices, mean_vectors, std_vectors, tracking_name):
    """
    Project features into SVD space using precomputed SVD matrices.

    Args:
        features (np.ndarray): Shape (num_samples, 512, 3, 3)
        svd_matrices (dict): SVD matrices loaded from the file
        mean_vectors (dict): Mean vectors loaded from the file
        std_vectors (dict): Standard deviation vectors loaded from the file
        tracking_name (str): The tracking name to process

    Returns:
        np.ndarray: Projected features of shape (num_samples, 512, 3, 3)
    """
    features = np.array(object=features)
    top_features = 128
    projected_features = np.zeros((features.shape[0], top_features, features.shape[2], features.shape[3]))
    
    for i in range(3):
        for j in range(3):
            # Extract features for this grid position
            grid_features = features[:, :, i, j]

            # Normalize using mean and std
            mean_vector = mean_vectors[tracking_name][(i, j)]
            std_vector = std_vectors[tracking_name][(i, j)]
            normalized_features = (grid_features - mean_vector) / std_vector

            # Project using SVD
            projection_matrix = svd_matrices[tracking_name][(i, j)]
            reduced_projection_matrix = projection_matrix[:top_features, :]
            projected_grid_features = np.dot(normalized_features, reduced_projection_matrix.T)

            # Save the projected features back to the grid position
            projected_features[:, :, i, j] = projected_grid_features

    return projected_features


def track_nuscenes():
    
    parser = argparse.ArgumentParser(description="3D MULTI-OBJECT TRACKING based lidar and camera detected features.\n\
                                    You can hardcode other hyperparameteres if needed")

    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--dets_train', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/train_conv_layer51233_thr020_050_interpolated.pkl",
                        help='Path to detections, train split for train - val split for inference')
    parser.add_argument('--dets_val', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/val_conv_layer51233_thr020_050_interpolated.pkl",
                        help='Path to detections, train split for train - val split for inference')
    parser.add_argument('--svd_lidar', type=str,
                    default="/home/ktsiakas/thesis_new/PROB_3D_MULMOD_MOT/svd_matrices_spatial_51233.pkl",
                    help='SVD matrices for lower representation')
    parser.add_argument('--svd_cam', type=str,
                default="/home/ktsiakas/thesis_new/PROB_3D_MULMOD_MOT/svd_matrices_cam.pkl",
                help='SVD matrices for lower representation')
    
    parser.add_argument('--training', type=str, default=True,
                        help='True or False not in ' '')

    parser.add_argument('--load_model_state', type=str, default='g4_train_weight_nonmatc_div_match.pth',
                        help='destination and name for model to load (for state == 0 leave as default)')
    parser.add_argument('--save_model_state', type=str, default='g4_train_weight_nonmatc_div_match.pth',
                        help='destination and name for model to save')
    parser.add_argument('--output_path', type=str, default='g4_train_weight_nonmatc_div_match.json',
                        help='destination for tracking results')

    args = parser.parse_args()

    # for nuscenes
    data_root = args.data_root
    version = args.version

    # data split
    dets_train = args.dets_train
    dets_val = args.dets_val

    # internal states of model
    training = args.training

    # load and save model DO THIS LATER
    load_model_state = args.load_model_state
    save_model_state = args.save_model_state

    output_path = args.output_path

    Tracker = TrackerNN().to(device)

    if not training:
        Tracker = load_tracker_states(Tracker, load_model_state)
        
    # Tracker = load_tracker_states(Tracker, load_model_state)

    params_to_optimize = list(Tracker.G1.parameters()) + list(Tracker.G4.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=0.001)
    Tracker.G1.train()
    Tracker.G4.train()
    criterion = nn.BCELoss(reduction='none')

    if training == True:
        EPOCHS = 11
    else:
        EPOCHS = 1


    # LOAD DATA TRAIN
    with open(dets_train, 'rb') as f:
        all_results = pickle.load(f)

    with open(args.svd_lidar, 'rb') as f:
        svd_data = pickle.load(f)

    with open(args.svd_cam, 'rb') as f:
        svd_data_cam = pickle.load(f)


    svd_matrices = svd_data['svd_matrices']
    mean_vectors = svd_data['mean_vectors']
    std_vectors = svd_data['std_vectors']

    svd_matrices_cam = svd_data_cam['svd_matrices']
    mean_vectors_cam = svd_data_cam['mean_vectors']
    std_vectors_cam = svd_data_cam['std_vectors']

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    total_time = 0.0
    total_frames = 0

    # writer = SummaryWriter('runs/empty')

    for epoch in range(EPOCHS):

        results = {}

        print('epoch', epoch + 1)

        epoch_loss = None

        if epoch == EPOCHS - 1:
            training = False
            with open(dets_val, 'rb') as f:
                all_results = pickle.load(f)

            Tracker.G1.eval()
            for param in Tracker.G1.parameters():
                param.requires_grad = False
            Tracker.G4.eval()
            for param in Tracker.G4.parameters():
                param.requires_grad = False

        processed_scene_tokens = set()

        for sample, sample_data in tqdm(all_results.items()):
            
            scene_token = nusc.get('sample', sample)['scene_token']

            if scene_token in processed_scene_tokens:
                continue

            first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
            current_sample_token = first_sample_token

            for tracking_name in NUSCENES_TRACKING_NAMES:
                Tracker.reinit_ab3dmot(tracking_name=tracking_name, training=training, criterion=criterion, epoch=epoch)

            prev_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            current_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            prev_trackers = {}

            # optimizer.zero_grad()
            # scene_loss = None

            while current_sample_token != '':
                
                current_ground_truths = create_box_annotations(current_sample_token, nusc)
                
                previous_sample_token = nusc.get('sample', current_sample_token)['prev']

                if previous_sample_token:
                    prev_ground_truths = create_box_annotations(previous_sample_token, nusc)
                else:
                    prev_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}


                results[current_sample_token] = []

                dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                fvecs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                pcbs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                cam_vecs = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

                for i, item in enumerate(all_results[current_sample_token]):
                    for name in NUSCENES_TRACKING_NAMES:
                        used_gt_indices = set()
                        for dets_outputs in item[name]:
                            dets[name].append(dets_outputs['box'])
                            pcbs[name].append(dets_outputs['point_cloud_features'])
                            fvecs[name].append(dets_outputs['feature_vector'])
                            cam_vecs[name].append(dets_outputs['camera_onehot_vector'])
                            info[name].append(dets_outputs['pred_score'])

                            # if dets_outputs['pred_score'] > 0.65:
                            #     dets[name].append(dets_outputs['box'])
                            #     pcbs[name].append(dets_outputs['point_cloud_features'])
                            #     fvecs[name].append(dets_outputs['feature_vector'])
                            #     cam_vecs[name].append(dets_outputs['camera_onehot_vector'])
                            #     info[name].append(dets_outputs['pred_score'])


                            # det_coords = np.array(dets_outputs['box'][:2])

                            # # Find the closest unmatched ground truth
                            # closest_gt_index = None
                            # min_distance = float('inf')

                            # for gt_idx, gt in enumerate(current_ground_truths[name]):
                            #     if gt_idx in used_gt_indices:  # Skip already used ground truths
                            #         continue

                            #     gt_coords = np.array(gt[:2], dtype=float)
                            #     distance = np.linalg.norm(det_coords - gt_coords)

                            #     if distance < 2.0 and distance < min_distance:
                            #         closest_gt_index = gt_idx
                            #         min_distance = distance

                            # # If a match is found, lock the ground truth and process
                            # if closest_gt_index is not None:
                            #     used_gt_indices.add(closest_gt_index)  # Mark this ground truth as used
                            #     dets[name].append(dets_outputs['box'])
                            #     pcbs[name].append(dets_outputs['point_cloud_features'])
                            #     fvecs[name].append(dets_outputs['feature_vector'])
                            #     cam_vecs[name].append(dets_outputs['camera_onehot_vector'])
                            #     info[name].append(dets_outputs['pred_score'])
                # # projection
                # for name in NUSCENES_TRACKING_NAMES:
                #     if len(pcbs[name]) == 0:
                #         continue

                #     pcbs[name] = project_to_svd_space(
                #     pcbs[name],
                #     svd_matrices,
                #     mean_vectors,
                #     std_vectors,
                #     name)           

                #     mean_vector = mean_vectors_cam[name]
                #     std_vector = std_vectors_cam[name]
                #     projection_matrix = svd_matrices_cam[name]

                #     features = np.array([feature.flatten() for feature in fvecs[name]])  # Shape: (num_features, 4608)

                #     # Mean-shift and normalize all features
                #     normalized_features = (features - mean_vector) / std_vector  # Shape: (num_features, 4608)

                #     # Project all features to lower dimensions
                #     reduced_projection_matrix = projection_matrix[:128, :]
                #     projected_features = np.dot(normalized_features, reduced_projection_matrix.T)
                #     # projected_features = projected_features.reshape(-1, 8, 3, 3)
                #     fvecs[name] = projected_features

                dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]),
                                            'pcbs': np.array(pcbs[tracking_name]),
                                            'fvecs': np.array(fvecs[tracking_name]),
                                            'cam_vecs': np.array(cam_vecs[tracking_name]),
                                            'info': np.array(info[tracking_name]),
                                            'current_gts': np.array(current_ground_truths[tracking_name]),
                                            'previous_gts': np.array(prev_ground_truths[tracking_name])
                                            }
                            for tracking_name in NUSCENES_TRACKING_NAMES}

                total_frames += 1
                start_time = time.time()

                optimizer.zero_grad()
                sample_loss = None
                
                # train
                cs = 0.0
                if epoch < EPOCHS - 1:
                    for tracking_name in NUSCENES_TRACKING_NAMES:
                        if dets_all[tracking_name]['dets'].shape[0] > 0:
                            
                            cs = 1.0 + cs
                            # optimizer.zero_grad()
                            trackers, loss = Tracker.forward(dets_all[tracking_name], tracking_name)

                            # if loss is not None:
                            #     loss.backward()
                            #     torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                            #     optimizer.step()

                            if loss is not None:
                                # if scene_loss is None:
                                if sample_loss is None:  
                                    sample_loss = loss
                                    # scene_loss = loss
                                else:
                                    sample_loss = sample_loss + loss
                                    # scene_loss = scene_loss + loss

                # val
                if epoch == EPOCHS - 1:

                    for tracking_name in NUSCENES_TRACKING_NAMES:
                        if dets_all[tracking_name]['dets'].shape[0] > 0:
                            with torch.no_grad():
                                trackers, loss = Tracker.forward(dets_all[tracking_name], tracking_name)

                            # (N, 9)
                            # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
                            for i in range(trackers.shape[0]):
                                sample_result, prev_trackers = format_sample_result(current_sample_token, tracking_name,
                                                                                    trackers[i],
                                                                                    prev_trackers)
                                results[current_sample_token].append(sample_result)

                cycle_time = time.time() - start_time
                total_time += cycle_time

                if epoch < EPOCHS - 1 and sample_loss is not None:
                    sample_loss.div(cs)
                    cs = 0
                    sample_loss.backward()
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                    optimizer.step()
                    # epoch_loss = epoch_loss + sample_loss.detach()

                # prev_ground_truths = copy.deepcopy(current_ground_truths)
                current_sample_token = nusc.get('sample', current_sample_token)['next']

            # if epoch < EPOCHS - 1 and scene_loss is not None:
            #     scene_loss.div(40.0)
            #     scene_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            #     optimizer.step()
                
                # epoch_loss = epoch_loss + scene_token


            processed_scene_tokens.add(scene_token)
            Tracker.clear_tracking_states()


        print("Total learning took: %.3f for %d frames or %.1f FPS" % (
            total_time, total_frames, total_frames / total_time))

        # writer.add_scalar('Loss/total', epoch_loss.item(), epoch)

        # scheduler.step(epoch_loss.item())

    #     for name, param in Tracker.G1.named_parameters():
    #         if param.grad is not None:
    #             writer.add_scalar(f'Gradients_G1/{name}/norm', param.grad.norm().item(), epoch)
    #             writer.add_histogram(f'Gradients_G1/{name}/histogram', param.grad, epoch)
    #             writer.add_histogram(f'Weights_G1/{name}/histogram', param.data, epoch)
    #             writer.add_scalar(f'Gradients_G1/Mean/{name}', param.grad.mean().item(), epoch)
    #             writer.add_scalar(f'Gradients_G1/Std/{name}', param.grad.std().item(), epoch)

    #     for name, param in Tracker.G2.named_parameters():
    #         if param.grad is not None:
    #             writer.add_scalar(f'Gradients_G2/{name}/norm', param.grad.norm().item(), epoch)
    #             writer.add_histogram(f'Gradients_G2/{name}/histogram', param.grad, epoch)
    #             writer.add_histogram(f'Weights_G2/{name}/histogram', param.data, epoch)
    #             writer.add_scalar(f'Gradients_G2/Mean/{name}', param.grad.mean().item(), epoch)
    #             writer.add_scalar(f'Gradients_G2/Std/{name}', param.grad.std().item(), epoch)

    #     for name, param in Tracker.G3.named_parameters():
    #         if param.grad is not None:
    #             writer.add_scalar(f'Gradients_G3/{name}/norm', param.grad.norm().item(), epoch)
    #             writer.add_histogram(f'Gradients_G3/{name}/histogram', param.grad, epoch)
    #             writer.add_histogram(f'Weights_G3/{name}/histogram', param.data, epoch)
    #             writer.add_scalar(f'Gradients_G3/Mean/{name}', param.grad.mean().item(), epoch)
    #             writer.add_scalar(f'Gradients_G3/Std/{name}', param.grad.std().item(), epoch)

    #     for name, param in Tracker.G4.named_parameters():
    #         if param.grad is not None:
    #             writer.add_scalar(f'Gradients_G4/{name}/norm', param.grad.norm().item(), epoch)
    #             writer.add_histogram(f'Gradients_G4/{name}/histogram', param.grad, epoch)
    #             writer.add_histogram(f'Weights_G4/{name}/histogram', param.data, epoch)
    #             writer.add_scalar(f'Gradients_G4/Mean/{name}', param.grad.mean().item(), epoch)
    #             writer.add_scalar(f'Gradients_G4/Std/{name}', param.grad.std().item(), epoch)

    #     writer.flush()

    # writer.close()

    # # save tracking results after inference
    save_tracker_states(Tracker, save_model_state)

    meta = {
        "use_camera": True,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False
    }

    output_data = {'meta': meta, 'results': results}
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile)

    print('results .json saved as', output_path)

if __name__ == '__main__':
    print('SOMETHING CATCHY')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    
    track_nuscenes()