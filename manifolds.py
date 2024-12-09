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

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
    # 'false_detections'
]


def visualize_feature_distributions(name, pcds, fvs):
    # Visualize pcds feature distributions

    plt.figure(figsize=(12, 6))
    sns.histplot(pcds.flatten(), kde=True)
    plt.title(f'Feature distribution of pcds for {name}')
    plt.savefig(f'features/distributions/pcds_distribution_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

    # Visualize fvs feature distributions

    plt.figure(figsize=(12, 6))
    sns.histplot(fvs.flatten(), kde=True)
    plt.title(f'Feature distribution of fvs for {name}')
    plt.savefig(f'features/distributions/fvs_distribution_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory


def visualize_feature_correlation(name, pcds_all, fvs_all):

    pcds_corr = np.corrcoef(pcds_all, rowvar=False)
    fvs_corr = np.corrcoef(fvs_all, rowvar=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pcds_corr, cmap='coolwarm', annot=False)
    plt.title(f'Correlation Matrix of pcds for {name}')
    plt.savefig(f'features/correlation/pcds_correlation_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

    plt.figure(figsize=(10, 8))
    sns.heatmap(fvs_corr, cmap='coolwarm', annot=False)
    plt.title(f'Correlation Matrix of fvs for {name}')
    plt.savefig(f'features/correlation/fvs_correlation_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory


def visualize_pca(name, pcds_all, fvs_all):
    pca = PCA(n_components=2)

    pcds_pca = pca.fit_transform(pcds_all)
    fvs_pca = pca.fit_transform(fvs_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(pcds_pca[:, 0], pcds_pca[:, 1], alpha=0.5)
    plt.title(f'PCA of pcds for {name}')
    plt.savefig(f'features/pca/pcds_pca_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

    plt.figure(figsize=(8, 6))
    plt.scatter(fvs_pca[:, 0], fvs_pca[:, 1], alpha=0.5)
    plt.title(f'PCA of fvs for {name}')
    plt.savefig(f'features/pca/fvs_pca_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory


def visualize_pca_3d(name, pcds_all, fvs_all):

    pca = PCA(n_components=3)

    pcds_pca = pca.fit_transform(pcds_all)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcds_pca[:, 0], pcds_pca[:, 1], pcds_pca[:, 2], alpha=0.6)
    ax.set_title(f'3D PCA of pcds for {name}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig(f'features/pca/pcds_pca_3d_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

    fvs_pca = pca.fit_transform(fvs_all)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fvs_pca[:, 0], fvs_pca[:, 1], fvs_pca[:, 2], alpha=0.6)
    ax.set_title(f'3D PCA of fvs for {name}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig(f'features/pca/fvs_pca_3d_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory



def visualize_umap(name, pcds_all, fvs_all):
    # Set parameters with cosine metric
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', n_components=2)

    # Fit and transform pcds
    pcds_umap = reducer.fit_transform(pcds_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(pcds_umap[:, 0], pcds_umap[:, 1], alpha=0.5, label='pcds')
    plt.title(f'UMAP of pcds for {name} with Cosine Metric')
    plt.savefig(f'features/umap/umap_pcds_{name}_cosine.png')
    plt.close()

    # Fit and transform fvs
    fvs_umap = reducer.fit_transform(fvs_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(fvs_umap[:, 0], fvs_umap[:, 1], alpha=0.5, label='fvs')
    plt.title(f'UMAP of fvs for {name} with Cosine Metric')
    plt.savefig(f'features/umap/umap_fvs_{name}_cosine.png')
    plt.close()


def visualize_umap_3d(name, pcds_all, fvs_all):
    reducer = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.1, metric='euclidean')

    pcds_umap = reducer.fit_transform(pcds_all)
    fvs_umap = reducer.fit_transform(fvs_all)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcds_umap[:, 0], pcds_umap[:, 1], pcds_umap[:, 2], alpha=0.6)
    ax.set_title(f'3D UMAP of pcds for {name}')
    plt.savefig(f'features/umap3d/umap_3d_pcds_{name}.png')
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fvs_umap[:, 0], fvs_umap[:, 1], fvs_umap[:, 2], alpha=0.6)
    ax.set_title(f'3D UMAP of fvs for {name}')
    plt.savefig(f'features/umap3d/umap_3d_fvs_{name}.png')
    plt.close()


def visualize_isomap_3d(name, pcds_all, fvs_all):
    n_neighbors = 20

    while True:
        try:
            # Convert data to lil_matrix to improve efficiency
            pcds_lil = lil_matrix(pcds_all)
            iso_pcds = Isomap(n_components=3, n_neighbors=n_neighbors)
            pcds_iso = iso_pcds.fit_transform(pcds_lil)

            fvs_lil = lil_matrix(fvs_all[name])
            iso_fvs = Isomap(n_components=3, n_neighbors=n_neighbors)
            fvs_iso = iso_fvs.fit_transform(fvs_lil)

            break  # Exit the loop if Isomap completes successfully

        except UserWarning as e:
            if "connected components" in str(e):
                n_neighbors += 5  # Increase the number of neighbors if the graph is disconnected
                print(f"Increasing neighbors to {n_neighbors} for {name}")
            else:
                raise e

    # Plot and save Isomap results for pcds
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcds_iso[:, 0], pcds_iso[:, 1], pcds_iso[:, 2], alpha=0.6)
    ax.set_title(f'3D Isomap of pcds for {name}')
    plt.savefig(f'features/isomap/isomap_3d_pcds_{name}.png')
    plt.close()

    # Plot and save Isomap results for fvs
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fvs_iso[:, 0], fvs_iso[:, 1], fvs_iso[:, 2], alpha=0.6)
    ax.set_title(f'3D Isomap of fvs for {name}')
    plt.savefig(f'features/isomap/isomap_3d_fvs_{name}.png')
    plt.close()


def visualize_tsne_3d(name, pcds_all, fvs_all):
    tsne = TSNE(n_components=3, random_state=42)

    pcds_tsne = tsne.fit_transform(pcds_all)
    fvs_tsne = tsne.fit_transform(fvs_all)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcds_tsne[:, 0], pcds_tsne[:, 1], pcds_tsne[:, 2], alpha=0.6)
    ax.set_title(f'3D t-SNE of pcds for {name}')
    plt.savefig(f'features/tsne/tsne_3d_pcds_{name}.png')
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fvs_tsne[:, 0], fvs_tsne[:, 1], fvs_tsne[:, 2], alpha=0.6)
    ax.set_title(f'3D t-SNE of fvs for {name}')
    plt.savefig(f'features/tsne/tsne_3d_fvs_{name}.png')
    plt.close()


def visualize_lda_3d(pcds_all, fvs_all):

    all_labels = []
    
    for name in NUSCENES_TRACKING_NAMES:
        all_labels.extend([name] * len(pcds_all[name]))

    # Encode class names to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)

    lda = LinearDiscriminantAnalysis(n_components=3)

    pcds_lda = lda.fit_transform(pcds_all, encoded_labels)
    fvs_lda = lda.fit_transform(fvs_all, encoded_labels)

    fig = plt.figure(figsize=(18, 12))  # Adjust figure size for more subplots

    # PCDs Views
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(pcds_lda[:, 0], pcds_lda[:, 1], pcds_lda[:, 2], c=encoded_labels, cmap='viridis', alpha=0.6)
    ax1.set_title('3D LDA of PCDs - View 1')
    ax1.set_xlabel('LDA Component 1')
    ax1.set_ylabel('LDA Component 2')
    ax1.set_zlabel('LDA Component 3')

    ax2 = fig.add_subplot(232, projection='3d')
    scatter2 = ax2.scatter(pcds_lda[:, 0], pcds_lda[:, 2], pcds_lda[:, 1], c=encoded_labels, cmap='viridis', alpha=0.6)
    ax2.set_title('3D LDA of PCDs - View 2')
    ax2.set_xlabel('LDA Component 1')
    ax2.set_ylabel('LDA Component 3')
    ax2.set_zlabel('LDA Component 2')

    ax3 = fig.add_subplot(233, projection='3d')
    scatter3 = ax3.scatter(pcds_lda[:, 1], pcds_lda[:, 2], pcds_lda[:, 0], c=encoded_labels, cmap='viridis', alpha=0.6)
    ax3.set_title('3D LDA of PCDs - View 3')
    ax3.set_xlabel('LDA Component 2')
    ax3.set_ylabel('LDA Component 3')
    ax3.set_zlabel('LDA Component 1')

    # FVs Views
    ax4 = fig.add_subplot(234, projection='3d')
    scatter4 = ax4.scatter(fvs_lda[:, 0], fvs_lda[:, 1], fvs_lda[:, 2], c=encoded_labels, cmap='viridis', alpha=0.6)
    ax4.set_title('3D LDA of FVs - View 1')
    ax4.set_xlabel('LDA Component 1')
    ax4.set_ylabel('LDA Component 2')
    ax4.set_zlabel('LDA Component 3')

    ax5 = fig.add_subplot(235, projection='3d')
    scatter5 = ax5.scatter(fvs_lda[:, 0], fvs_lda[:, 2], fvs_lda[:, 1], c=encoded_labels, cmap='viridis', alpha=0.6)
    ax5.set_title('3D LDA of FVs - View 2')
    ax5.set_xlabel('LDA Component 1')
    ax5.set_ylabel('LDA Component 3')
    ax5.set_zlabel('LDA Component 2')

    ax6 = fig.add_subplot(236, projection='3d')
    scatter6 = ax6.scatter(fvs_lda[:, 1], fvs_lda[:, 2], fvs_lda[:, 0], c=encoded_labels, cmap='viridis', alpha=0.6)
    ax6.set_title('3D LDA of FVs - View 3')
    ax6.set_xlabel('LDA Component 2')
    ax6.set_ylabel('LDA Component 3')
    ax6.set_zlabel('LDA Component 1')

    plt.colorbar(scatter1, ax=ax1, label='Classes')
    plt.colorbar(scatter4, ax=ax4, label='Classes')

    plt.tight_layout()  # Adjust subplots to fit in the figure area
    plt.savefig('features/lda/lda_3d_all_views.png')
    plt.close()


def process_category(name, pcds, fvs):

    # visualize_pca(name, pcds, fvs)
    # visualize_pca_3d(name, pcds, fvs)

    # visualize_umap(name, pcds, fvs)
    visualize_umap_3d(name, pcds, fvs)
    
    # visualize_tsne_3d(name, pcds, fvs)



    # visualize_lda_3d(name, pcds, fvs)
    # visualize_isomap_3d(pcds_all, fvs_all)


def visualize_features_multiprocessing(pcds_all, fvs_all):

    for name in NUSCENES_TRACKING_NAMES:

        pcds = pcds_all[name].reshape(len(pcds_all[name]), -1)
        process_category(name, pcds, fvs_all[name])


def create_box_annotations(sample_token,nusc):
    ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

    sample = nusc.get('sample', sample_token)
    ann_token = sample['anns']

    for ann in ann_token:

        ann_meta = nusc.get('sample_annotation', ann)
        t_name = ann_meta['category_name']

        for tracking_name in NUSCENES_TRACKING_NAMES:
            if tracking_name in t_name:

                trs = np.array(ann_meta['translation'])

                ground_truths[tracking_name].append(trs)

    return ground_truths


def feature_analysis():
    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--data', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val_2.pkl",
                        help='Path to detections, train split for train - val split for inference')
    
    parser.add_argument('--method', type=str,
                    default=0,
                    help='0 distance based, 1 threshold based')

    parser.add_argument('--distance', type=str,
                    default=2,
                    help='Distance to groundtruths based denoising')
    
    parser.add_argument('--keep', type=str,
                    default=True,
                    help='True == Keep detections below distance threshold')
    
    parser.add_argument('--thresh_1', type=str,
                    default=0.1,
                    help='Lower threshold bound acceptance of detection')
    
    parser.add_argument('--thresh_2', type=str,
                    default=0.95,
                    help='Upper threshold bound acceptance of detection')
    
    args = parser.parse_args()
    data = args.data
    data_root = args.data_root
    version = args.version

    # os.makedirs('features/distributions', exist_ok=True)
    # os.makedirs('features/statistics', exist_ok=True)
    # os.makedirs('features/correlation', exist_ok=True)
    os.makedirs('features/pca', exist_ok=True)
    # os.makedirs('features/density', exist_ok=True)
    os.makedirs('features/umap', exist_ok=True)
    os.makedirs('features/class_imbalance', exist_ok=True)
    os.makedirs('features/umap3d', exist_ok=True)
    os.makedirs('features/tsne', exist_ok=True)
    # os.makedirs('features/lda', exist_ok=True)

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
                    for dets_outputs in item[name]:

                        if args.method == 0:
                            det_coords = np.array(dets_outputs['box'][:2])

                            for gt_coords in current_ground_truths[name]:
                                gt_coords = np.array(gt_coords[:2])
                                distance = np.linalg.norm(det_coords - gt_coords)

                                if args.keep:
                                    if distance < args.distance:
                                        pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                                        pcds_all[name].append(pcd_feature)
                                        fvs_all[name].append(dets_outputs['feature_vector'])
                                else:
                                    if distance >= args.distance:
                                        pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                                        pcds_all[name].append(pcd_feature)
                                        fvs_all[name].append(dets_outputs['feature_vector'])

                        else:
                            if dets_outputs['pred_score'] > args.thresh_1 and dets_outputs['pred_score'] < args.thresh_2:
                                pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                                pcds_all[name].append(pcd_feature)
                                fvs_all[name].append(dets_outputs['feature_vector'])

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

    for name in NUSCENES_TRACKING_NAMES:

        plt.figure(figsize=(12, 6))
        plt.bar(object_counts.keys(), object_counts.values())
        plt.xlabel('Tracking Names')
        plt.ylabel('Number of Objects')
        plt.title('Number of Objects for Each Tracking Name in pcds_all')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('features/class_imbalance/object_counts_histogram.png')
        plt.close()

    visualize_features_multiprocessing(pcds_all, fvs_all)


if __name__ == '__main__':
    feature_analysis()
