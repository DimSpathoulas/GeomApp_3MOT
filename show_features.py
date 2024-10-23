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
    'truck'
]


def visualize_feature_distributions(pcds_all, fvs_all):
    # Visualize pcds feature distributions
    for name in NUSCENES_TRACKING_NAMES:
        plt.figure(figsize=(12, 6))
        sns.histplot(pcds_all[name].flatten(), kde=True)
        plt.title(f'Feature distribution of pcds for {name}')
        plt.savefig(f'features/distributions/pcds_distribution_{name}.png')  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory

    # Visualize fvs feature distributions
    for name in NUSCENES_TRACKING_NAMES:
        plt.figure(figsize=(12, 6))
        sns.histplot(fvs_all[name].flatten(), kde=True)
        plt.title(f'Feature distribution of fvs for {name}')
        plt.savefig(f'features/distributions/fvs_distribution_{name}.png')  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory


def visualize_feature_statistics(name, pcds_all, fvs_all):
    # Calculate the mean and variance of pcds
    pcds_mean = np.mean(pcds_all, axis=0)  # Shape (512, 3, 3)
    pcds_var = np.var(pcds_all, axis=0)  # Shape (512, 3, 3)

    # If you want to plot the mean and variance, you might want to flatten the variance
    pcds_var_flat = pcds_var.reshape(pcds_var.shape[0], -1)  # Shape (512, 9)

    # Calculate a single variance value per feature if needed, for example, by taking the mean across the last dimensions
    pcds_var_mean = np.mean(pcds_var_flat, axis=1)  # Shape (512,)

    # Plot mean and variance
    plt.figure(figsize=(12, 6))
    plt.plot(pcds_mean.flatten(), label='Mean of pcds')
    plt.plot(pcds_var_mean, label='Mean Variance of pcds', linestyle='--')
    plt.title(f'Mean and Variance of pcds for {name}')
    plt.legend()
    plt.savefig(f'features/statistics/pcds_statistics_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

    # Repeat similar steps for fvs
    fvs_mean = np.mean(fvs_all[name], axis=0)  # Shape (1024,)
    fvs_var = np.var(fvs_all[name], axis=0)  # Shape (1024,)

    plt.figure(figsize=(12, 6))
    plt.plot(fvs_mean, label='Mean of fvs')
    plt.plot(fvs_var, label='Variance of fvs', linestyle='--')
    plt.title(f'Mean and Variance of fvs for {name}')
    plt.legend()
    plt.savefig(f'features/statistics/fvs_statistics_{name}.png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory


def visualize_feature_correlation(name, pcds_all, fvs_all):

    pcds_reshaped = pcds_all.reshape(pcds_all.shape[0], -1)
    pcds_corr = np.corrcoef(pcds_reshaped, rowvar=False)
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

    pcds_reshaped = pcds_all.reshape(pcds_all.shape[0], -1)
    pcds_pca = pca.fit_transform(pcds_reshaped)
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


def visualize_pca_3d(pcds_all, fvs_all):
    for name in NUSCENES_TRACKING_NAMES:
        pca = PCA(n_components=3)

        # fvs_reshaped = fvs_all[name].reshape(fvs_all[name].shape[0], -1)
        # fvs_pca = pca.fit_transform(fvs_reshaped)

        pcds_reshaped = pcds_all[name].reshape(pcds_all[name].shape[0], -1)
        pcds_pca = pca.fit_transform(pcds_reshaped)

        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(fvs_pca[:, 0], fvs_pca[:, 1], fvs_pca[:, 2], alpha=0.6)
        # ax.set_title(f'3D PCA of fvs for {name}')
        # ax.set_xlabel('Principal Component 1')
        # ax.set_ylabel('Principal Component 2')
        # ax.set_zlabel('Principal Component 3')
        # plt.savefig(f'features/pca/fvs_pca_3d_{name}.png')  # Save the plot as a PNG file
        # plt.close()  # Close the plot to free memory

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcds_pca[:, 0], pcds_pca[:, 1], pcds_pca[:, 2], alpha=0.6)
        ax.set_title(f'3D PCA of pcds for {name}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.savefig(f'features/pca/pcds_pca_3d_{name}.png')  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory


def visualize_density_estimation(pcds_all, fvs_all):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)

    for name in NUSCENES_TRACKING_NAMES:
        pcds_reshaped = pcds_all[name].reshape(pcds_all[name].shape[0], -1)
        kde.fit(pcds_reshaped)
        pcds_samples = kde.sample(1000)

        kde.fit(fvs_all[name])
        fvs_samples = kde.sample(1000)

        # KDE for point cloud descriptors (pcds)
        plt.figure(figsize=(8, 6))
        sns.kdeplot(np.vstack(pcds_samples).T)  # Transpose to create 2D data for KDE
        plt.title(f'KDE of pcds for {name}')
        plt.savefig(f'features/density/pcds_kde_{name}.png')  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory

        # KDE for feature vectors (fvs)
        plt.figure(figsize=(8, 6))
        sns.kdeplot(np.vstack(fvs_samples).T)  # Transpose to create 2D data for KDE
        plt.title(f'KDE of fvs for {name}')
        plt.savefig(f'features/density/fvs_kde_{name}.png')  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory


def visualize_umap(pcds_all, fvs_all):
    for name in NUSCENES_TRACKING_NAMES:
        # Set parameters with cosine metric
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', n_components=2)

        # Fit and transform pcds
        pcds_umap = reducer.fit_transform(pcds_all[name].reshape(len(pcds_all[name]), -1))

        plt.figure(figsize=(8, 6))
        plt.scatter(pcds_umap[:, 0], pcds_umap[:, 1], alpha=0.5, label='pcds')
        plt.title(f'UMAP of pcds for {name} with Cosine Metric')
        plt.savefig(f'features/umap/umap_pcds_{name}_cosine.png')
        plt.close()

        # Fit and transform fvs
        fvs_umap = reducer.fit_transform(fvs_all[name])

        plt.figure(figsize=(8, 6))
        plt.scatter(fvs_umap[:, 0], fvs_umap[:, 1], alpha=0.5, label='fvs')
        plt.title(f'UMAP of fvs for {name} with Cosine Metric')
        plt.savefig(f'features/umap/umap_fvs_{name}_cosine.png')
        plt.close()


def visualize_umap_3d(pcds_all, fvs_all):
    for name in NUSCENES_TRACKING_NAMES:
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine')

        pcds_umap = reducer.fit_transform(pcds_all[name].reshape(len(pcds_all[name]), -1))
        fvs_umap = reducer.fit_transform(fvs_all[name])

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


def visualize_isomap_3d(pcds_all, fvs_all):
    for name in NUSCENES_TRACKING_NAMES:
        n_neighbors = 20

        while True:
            try:
                # Convert data to lil_matrix to improve efficiency
                pcds_lil = lil_matrix(pcds_all[name].reshape(len(pcds_all[name]), -1))
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


def visualize_tsne_3d(pcds_all, fvs_all):
    for name in NUSCENES_TRACKING_NAMES:
        tsne = TSNE(n_components=3, random_state=42)

        pcds_tsne = tsne.fit_transform(pcds_all[name].reshape(len(pcds_all[name]), -1))
        fvs_tsne = tsne.fit_transform(fvs_all[name])

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


# def visualize_lda_3d(pcds_all, fvs_all):

#     all_pcds = []
#     all_fvs = []
#     all_labels = []

#     for i, name in enumerate(NUSCENES_TRACKING_NAMES):
#         all_pcds.append(pcds_all[name].reshape(len(pcds_all[name]), -1))
#         all_fvs.append(fvs_all[name])
#         all_labels.extend([i] * len(pcds_all[name]))

#     all_pcds = np.vstack(all_pcds)
#     all_fvs = np.vstack(all_fvs)
#     all_labels = np.array(all_labels)

#     lda = LinearDiscriminantAnalysis(n_components=3)

#     pcds_lda = lda.fit_transform(all_pcds, all_labels)
#     fvs_lda = lda.fit_transform(all_fvs, all_labels)

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter(pcds_lda[:, 0], pcds_lda[:, 1], pcds_lda[:, 2], c=all_labels, cmap='viridis', alpha=0.6)
#     ax.set_title('3D LDA of pcds for all classes')
#     plt.colorbar(scatter)
#     plt.savefig('features/lda/lda_3d_pcds_all.png')
#     plt.close()

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter(fvs_lda[:, 0], fvs_lda[:, 1], fvs_lda[:, 2], c=all_labels, cmap='viridis', alpha=0.6)
#     ax.set_title('3D LDA of fvs for all classes')
#     plt.colorbar(scatter)
#     plt.savefig('features/lda/lda_3d_fvs_all.png')
#     plt.close()

def visualize_lda_3d(pcds_all, fvs_all):
    all_pcds = []
    all_fvs = []
    all_labels = []

    for name in NUSCENES_TRACKING_NAMES:
        all_pcds.append(pcds_all[name].reshape(len(pcds_all[name]), -1))
        all_fvs.append(fvs_all[name])
        all_labels.extend([name] * len(pcds_all[name]))  # Use class names instead of numbers

    all_pcds = np.vstack(all_pcds)
    all_fvs = np.vstack(all_fvs)

    # Encode class names to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)

    lda = LinearDiscriminantAnalysis(n_components=3)

    pcds_lda = lda.fit_transform(all_pcds, encoded_labels)
    fvs_lda = lda.fit_transform(all_fvs, encoded_labels)

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
    visualize_feature_statistics(name, pcds, fvs)
    visualize_feature_correlation(name, pcds, fvs)
    visualize_pca(name, pcds, fvs)


def visualize_features_multiprocessing(pcds_all, fvs_all):
    processes = []
    for name in NUSCENES_TRACKING_NAMES:
        p = mp.Process(target=process_category, args=(name, pcds_all[name], fvs_all[name]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def feature_analysis():
    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help='Root directory of the NuScenes dataset')
    parser.add_argument('--data', type=str,
                        default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val_2.pkl",
                        help='Path to detections, train split for train - val split for inference')

    args = parser.parse_args()
    data = args.data
    data_root = args.data_root
    version = args.version

    os.makedirs('features/distributions', exist_ok=True)
    os.makedirs('features/statistics', exist_ok=True)
    os.makedirs('features/correlation', exist_ok=True)
    os.makedirs('features/pca', exist_ok=True)
    os.makedirs('features/density', exist_ok=True)
    os.makedirs('features/umap', exist_ok=True)
    os.makedirs('features/class_imbalance', exist_ok=True)
    os.makedirs('features/umap3d', exist_ok=True)
    # os.makedirs('features/isomap', exist_ok=True)
    os.makedirs('features/tsne', exist_ok=True)
    os.makedirs('features/lda', exist_ok=True)

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

        while current_sample_token != '':

            for i, item in enumerate(all_results[current_sample_token]):
                for name in NUSCENES_TRACKING_NAMES:
                    for dets_outputs in item[name]:
                        pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                        pcds_all[name].append(pcd_feature)
                        fvs_all[name].append(dets_outputs['feature_vector'])

            current_sample_token = nusc.get('sample', current_sample_token)['next']

        processed_scene_tokens.add(scene_token)

    for name in NUSCENES_TRACKING_NAMES:
        pcds_all[name] = np.vstack(pcds_all[name])
        fvs_all[name] = np.vstack(fvs_all[name])

    print(f'Number of processed scene tokens: {len(processed_scene_tokens)}')

    for name in NUSCENES_TRACKING_NAMES:
        print(f'Shape of pcds_all[{name}]: {pcds_all[name].shape}')
        print(f'Shape of fvs_all[{name}]: {fvs_all[name].shape}')

    # object_counts = {name: pcds_all[name].shape[0] for name in NUSCENES_TRACKING_NAMES}

    # for name in NUSCENES_TRACKING_NAMES:
    #     print(f'Shape of pcds_all[{name}]: {pcds_all[name].shape}')

    #     plt.figure(figsize=(12, 6))
    #     plt.bar(object_counts.keys(), object_counts.values())
    #     plt.xlabel('Tracking Names')
    #     plt.ylabel('Number of Objects')
    #     plt.title('Number of Objects for Each Tracking Name in pcds_all')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig('features/class_imbalance/object_counts_histogram.png')
    #     plt.close()

    # visualize_feature_distributions(pcds_all, fvs_all)
    # print('done with 1')

    # visualize_feature_statistics(pcds_all, fvs_all)
    # print('done with 2')

    # visualize_feature_correlation(pcds_all, fvs_all)
    # print('done with 3')

    # visualize_pca(pcds_all, fvs_all)
    # print('done with 4')

    # visualize_density_estimation(pcds_all, fvs_all)
    # print('done with 5')

    # visualize_pca_3d(pcds_all, fvs_all)
    # print('done with 6')

    # visualize_umap(pcds_all, fvs_all)
    # print('done with all')

    # visualize_umap_3d(pcds_all, fvs_all)
    # print('done with UMAP 3D')

    # visualize_isomap_3d(pcds_all, fvs_all)
    # print('done with Isomap 3D')

    # visualize_tsne_3d(pcds_all, fvs_all)
    # print('done with t-SNE 3D')

    # visualize_lda_3d(pcds_all, fvs_all)
    # print('done with LDA 3D')

    # visualize_features_multiprocessing(pcds_all, fvs_all)


if __name__ == '__main__':
    feature_analysis()
