from __future__ import print_function
from tqdm import tqdm
import os.path
import pickle
import argparse
from nuscenes import NuScenes
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import umap
import numpy as np
import multiprocessing as mp
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE, Isomap
from scipy.sparse import lil_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
from sklearn.preprocessing import normalize

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
    'false_detections'
]

# Color map for different tracking names
COLOR_MAP = {
    'bicycle': 'blue',
    'bus': 'green',
    'car': 'red',
    'motorcycle': 'purple',
    'pedestrian': 'orange',
    'trailer': 'brown',
    'truck': 'cyan',
    'false_detections': 'black'
}

def visualize_combined_manifolds(name, pcds_all, fvs_all):
    # Ensure CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pcds_flattened = {k: v.reshape(v.shape[0], -1) for k, v in pcds_all.items()}

    # Convert numpy arrays to torch tensors and move to GPU
    pcds_tensor = {k: torch.from_numpy(v).float().to(device) for k, v in pcds_flattened.items()}
    fvs_tensor = {k: torch.from_numpy(v).float().to(device) for k, v in fvs_all.items()}

    # Combine all data
    pcds_combined = torch.cat(list(pcds_tensor.values()), dim=0)
    fvs_combined = torch.cat(list(fvs_tensor.values()), dim=0)

    # Create labels for color coding
    labels = []
    for k, v in pcds_flattened.items():
        labels.extend([k] * v.shape[0])
    
    # Perform PCA for 3D visualization
    pca_3d_pcds = PCA(n_components=3)
    pcds_pca_3d = pca_3d_pcds.fit_transform(pcds_combined.cpu().numpy())

    pca_3d_fvs = PCA(n_components=3)
    fvs_pca_3d = pca_3d_fvs.fit_transform(fvs_combined.cpu().numpy())

    # Define viewpoints (elev, azim)
    viewpoints = [
        (20, 30),  # Viewpoint 1
        (60, 120), # Viewpoint 2
        (90, 180)  # Viewpoint 3
    ]

    # Create plots with different viewpoints
    for i, (elev, azim) in enumerate(viewpoints):
        fig = plt.figure(figsize=(16, 8))
        
        # 3D PCA for point cloud data
        ax1 = fig.add_subplot(121, projection='3d')
        for tracking_name in NUSCENES_TRACKING_NAMES:
            mask = np.array(labels) == tracking_name
            ax1.scatter(
                pcds_pca_3d[mask, 0], 
                pcds_pca_3d[mask, 1], 
                pcds_pca_3d[mask, 2], 
                c=COLOR_MAP[tracking_name], 
                label=tracking_name, 
                alpha=0.7
            )
        ax1.set_title(f'3D PCA of pcds for {name} - View {i+1}')
        ax1.set_xlabel('PCA1')
        ax1.set_ylabel('PCA2')
        ax1.set_zlabel('PCA3')
        ax1.view_init(elev=elev, azim=azim)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3D PCA for feature vectors
        ax2 = fig.add_subplot(122, projection='3d')
        for tracking_name in NUSCENES_TRACKING_NAMES:
            mask = np.array(labels) == tracking_name
            ax2.scatter(
                fvs_pca_3d[mask, 0], 
                fvs_pca_3d[mask, 1], 
                fvs_pca_3d[mask, 2], 
                c=COLOR_MAP[tracking_name], 
                label=tracking_name, 
                alpha=0.7
            )
        ax2.set_title(f'3D PCA of fvs for {name} - View {i+1}')
        ax2.set_xlabel('PCA1')
        ax2.set_ylabel('PCA2')
        ax2.set_zlabel('PCA3')
        ax2.view_init(elev=elev, azim=azim)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'manifolds/pca/combined_pca_3d_view_{i+1}_{name}.png', bbox_inches='tight')
        plt.show()

    ### t-SNE Visualization
    tsne = TSNE(n_components=3, random_state=42)
    normalized_features = normalize(pcds_combined.cpu().numpy(), axis=1, norm='l2')
    normalized_feats = normalize(fvs_combined.cpu().numpy(), axis=1, norm='l2')
    pcds_tsne_3d = tsne.fit_transform(normalized_features)
    fvs_tsne_3d = tsne.fit_transform(normalized_feats)

    # Create additional viewpoints
    view_angles = [(60, 120), (90, 180), (120, 240)]
    for elev, azim in view_angles:
        fig = plt.figure(figsize=(18, 12))

        # Point cloud 3D t-SNE
        ax1 = fig.add_subplot(121, projection='3d')
        for tracking_name in NUSCENES_TRACKING_NAMES:
            mask = np.array(labels) == tracking_name
            ax1.scatter(
                pcds_tsne_3d[mask, 0], 
                pcds_tsne_3d[mask, 1], 
                pcds_tsne_3d[mask, 2], 
                c=COLOR_MAP[tracking_name], 
                label=tracking_name, 
                alpha=0.7
            )
        ax1.set_title(f'3D t-SNE of pcds for {name}')
        ax1.set_xlabel('TSNE1')
        ax1.set_ylabel('TSNE2')
        ax1.set_zlabel('TSNE3')
        ax1.view_init(elev=elev, azim=azim)  # Set new viewpoint

        # Feature vector 3D t-SNE
        ax2 = fig.add_subplot(122, projection='3d')
        for tracking_name in NUSCENES_TRACKING_NAMES:
            mask = np.array(labels) == tracking_name
            ax2.scatter(
                fvs_tsne_3d[mask, 0], 
                fvs_tsne_3d[mask, 1], 
                fvs_tsne_3d[mask, 2], 
                c=COLOR_MAP[tracking_name], 
                label=tracking_name, 
                alpha=0.7
            )
        ax2.set_title(f'3D t-SNE of fvs for {name}')
        ax2.set_xlabel('TSNE1')
        ax2.set_ylabel('TSNE2')
        ax2.set_zlabel('TSNE3')
        ax2.view_init(elev=elev, azim=azim)  # Set new viewpoint

        plt.tight_layout()
        plt.savefig(f'manifolds/tsne/combined_tsne_3d_{name}_view_{elev}_{azim}.png', bbox_inches='tight')
        plt.close()


    # UMAP Visualization
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric='euclidean', n_components=3)
    
    pcds_umap_3d = reducer.fit_transform(pcds_combined.cpu().numpy())
    fvs_umap_3d = reducer.fit_transform(fvs_combined.cpu().numpy())

    # Create plots with different viewpoints
    view_angles = [(60, 120)]
    for elev, azim in view_angles:
        fig = plt.figure(figsize=(18, 12))
        
        # 3D UMAP for point cloud data
        ax1 = fig.add_subplot(121, projection='3d')
        for tracking_name in NUSCENES_TRACKING_NAMES:
            mask = np.array(labels) == tracking_name
            ax1.scatter(
                pcds_umap_3d[mask, 0], 
                pcds_umap_3d[mask, 1], 
                pcds_umap_3d[mask, 2], 
                c=COLOR_MAP[tracking_name], 
                label=tracking_name, 
                alpha=0.7
            )
        ax1.set_title(f'3D UMAP of pcds for {name}')
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.set_zlabel('UMAP3')
        ax1.view_init(elev=elev, azim=azim)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3D UMAP for feature vectors
        ax2 = fig.add_subplot(122, projection='3d')
        for tracking_name in NUSCENES_TRACKING_NAMES:
            mask = np.array(labels) == tracking_name
            ax2.scatter(
                fvs_umap_3d[mask, 0], 
                fvs_umap_3d[mask, 1], 
                fvs_umap_3d[mask, 2], 
                c=COLOR_MAP[tracking_name], 
                label=tracking_name, 
                alpha=0.7
            )
        ax2.set_title(f'3D UMAP of fvs for {name}')
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        ax2.set_zlabel('UMAP3')
        ax2.view_init(elev=elev, azim=azim)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'manifolds/umap/combined_umap_3d_view_{name}.png', bbox_inches='tight')
        plt.show()

def feature_analysis():
    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--data', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val.pkl",
                        help='Path to detections, train split for train - val split for inference')

    parser.add_argument('--distance', type=str,
                    default=2,
                    help='Distance to groundtruths based denoising')
    
    args = parser.parse_args()
    data = args.data
    data_root = args.data_root
    version = args.version

    os.makedirs('manifolds/pca', exist_ok=True)
    os.makedirs('manifolds/umap', exist_ok=True)
    os.makedirs('manifolds/class_imbalance', exist_ok=True)
    os.makedirs('manifolds/tsne', exist_ok=True)

    pcds_all = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}
    fvs_all = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    with open(data, 'rb') as f:
        all_results = pickle.load(f)

    processed_scene_tokens = set()

    for sample, sample_data in tqdm(all_results.items()):

        scene_token = nusc.get('sample', sample)['scene_token']

        if scene_token in processed_scene_tokens:
            continue

        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token

        current_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

        while current_sample_token != '':

            current_ground_truths = create_box_annotations(current_sample_token, nusc)

            for i, item in enumerate(all_results[current_sample_token]):
                for name in NUSCENES_TRACKING_NAMES:
                    if name != 'false_detections': 
                        used_gt_indices = set()  # Track indices of matched ground truths
                        unmatched_detections = []  # Track unmatched detections
                        
                        for dets_outputs in item[name]:
                            det_coords = np.array(dets_outputs['box'][:2])

                            # Find the closest unmatched ground truth
                            closest_gt_index = None
                            min_distance = float('inf')

                            for gt_idx, gt in enumerate(current_ground_truths[name]):
                                if gt_idx in used_gt_indices:  # Skip already used ground truths
                                    continue

                                gt_coords = np.array(gt['coords'][:2])
                                distance = np.linalg.norm(det_coords - gt_coords)

                                if distance < args.distance and distance < min_distance:
                                    closest_gt_index = gt_idx
                                    min_distance = distance

                            # If a match is found, lock the ground truth and process
                            if closest_gt_index is not None:
                                used_gt_indices.add(closest_gt_index)  # Mark this ground truth as used
                                pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                                pcds_all[name].append(pcd_feature)
                                fvs_all[name].append(dets_outputs['feature_vector'])
                            else:
                                # If no match is found, add to unmatched detections
                                unmatched_detections.append(dets_outputs)
                        
                        # Add unmatched detections to 'false_detections'
                        if unmatched_detections:
                            if 'false_detections' not in pcds_all:
                                pcds_all['false_detections'] = []
                                fvs_all['false_detections'] = []

                            for dets_outputs in unmatched_detections:
                                pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                                pcds_all['false_detections'].append(pcd_feature)
                                fvs_all['false_detections'].append(dets_outputs['feature_vector'])

            current_sample_token = nusc.get('sample', current_sample_token)['next']
            

            current_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

        processed_scene_tokens.add(scene_token)

    for name in NUSCENES_TRACKING_NAMES:
            pcds_all[name] = np.vstack(pcds_all[name])
            fvs_all[name] = np.vstack(fvs_all[name])

    print(f'Number of processed scene tokens: {len(processed_scene_tokens)}')

    for name in NUSCENES_TRACKING_NAMES:
        print(f'Shape of pcds_all[{name}]: {pcds_all[name].shape}')
        print(f'Shape of fvs_all[{name}]: {fvs_all[name].shape}')

    object_counts = {name: pcds_all[name].shape[0] for name in NUSCENES_TRACKING_NAMES}

    # Object count histogram
    plt.figure(figsize=(12, 6))
    plt.bar(object_counts.keys(), object_counts.values(), color=[COLOR_MAP[k] for k in object_counts.keys()])
    plt.xlabel('Tracking Names')
    plt.ylabel('Number of Objects')
    plt.title('Number of Objects for Each Tracking Name in pcds_all')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('manifolds/class_imbalance/object_counts_histogram.png')
    plt.close()

    # Visualize combined manifolds
    visualize_combined_manifolds('nuscenes', pcds_all, fvs_all)

def create_box_annotations(sample_token, nusc):
    # (Previous implementation remains the same)
    ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

    sample = nusc.get('sample', sample_token)
    ann_token = sample['anns']

    for ann in ann_token:
        ann_meta = nusc.get('sample_annotation', ann)
        t_name = ann_meta['category_name']

        for tracking_name in NUSCENES_TRACKING_NAMES:
            if tracking_name in t_name:
                trs = np.array(ann_meta['translation'])
                id = ann_meta['instance_token']
                ground_truths[tracking_name].append({'coords': trs, 'id': id})  # Include ID for later use

    return ground_truths

if __name__ == '__main__':
    feature_analysis()