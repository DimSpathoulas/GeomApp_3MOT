import numpy as np
import matplotlib.pyplot as plt

'''
ΠΡΟΒΛΗΜΑ:

ΔΕΝ ΜΟΥ ΕΓΚΥΑΤΑΙ ΚΑΝΕΙΣ ΟΤΙ Η ΜΙΚΡΟΤΕΡΗ ΑΠΟΣΤΑΣΗ ΕΙΝΑΙ ΚΑΙ ΑΥΤΗ ΠΟΥ ΕΧΕΙ ΙΔΙΟ INSTANCE TOKEN (ID)

ΛΥΣΗ:
ΨΑΧΝΩ ΟΛΑ ΤΑ ΚΟΝΤΙΝΑ GROUNDTRUTHS ΣΕ ΣΧΕΣΗ ΜΕ DETS-TRACKS (ΑΠΟ ΤΗΝ ΜΙΚΡΟΤΕΡΗ ΑΠΟΣΤΑΣΗ ΜΕΧΡΙ ΤΟ THRESHOLD)

ΟΛΙΚΗ ΑΠΟΣΤΑΣΗ:
ΑΠΟΣΤΑΣΗ ΑΠΟ DET CURR_GT + TRACK PREV_GT

ΠΡΟΒΛΗΜΑ: SUB-OPTIMALITY

ΔΕΝ ΜΟΥ ΕΓΚΥΑΤΑΙ ΚΑΝΕΙΣ ΟΤΙ ΚΑΝΩΝΤΑΣ ΑΥΤΟ ΤΟ ASSOCIATION ΔΕΝ ΔΙΑΓΡΑΦΩ ΚΑΠΟΙΟ ΕΠΟΜΕΝΟ ASSOCIATION
ΜΕ ΑΚΟΜΑ ΜΙΚΡΟΤΕΡΗ ΟΛΙΚΗ ΑΠΟΣΤΑΣΗ ΣΤΙΣ FOR

ΠΟΥ ΚΡΥΒΕΤΑΙ ΤΟ ΠΡΟΒΛΗΜΑ:
ΜΕΣΑ ΣΤΟΥΣ ΚΥΚΛΟΥΣ ΠΟΥ ΔΗΜΙΟΥΡΓΟΥΝΤΑΙ ΓΥΡΩ ΑΠΟ ΚΑΘΕ DET ΚΑΙ TRACK ΜΠΟΡΕΙ ΝΑ ΥΠΑΡΧΟΥΝ ΠΟΛΛΑΠΛΑ GTS ΜΕ ΙΔΙΑ ID.
ΘΕΛΩ ΜΕΣΑ ΑΠΟ ΟΛΟΥΣ ΤΟΥΣ ΣΥΝΔΙΑΣΜΟΥΣ ΝΑ ΒΡΩ ΤΟΝ OPTIMAL ΑΛΛΙΩΣ Ο Κ ΔΕΝ ΚΑΝΕΙ ΒΕΛΤΙΣΤΟ ASSOCIATION
ΔΕΝ ΜΠΟΡΕΙ ΝΑ ΛΥΘΕΙ ΜΕ ΑΠΛΟ ΚΑΤΩΦΛΙ ΜΙΝ ΓΙΑΤΙ ΒΡΙΣΚΩ ΠΑΛΙ ΤΗΝ ΕΛΑΧΙΣΤΗ ΑΠΟΣΤΑΣΗ
ΠΡΕΠΕΙ ΝΑ ΛΑΒΩ ΥΠΟΨΗΝ ΜΕ BRUTE FORCE ΑΝ ΧΡΕΙΑΣΤΕΙ ΟΛΟΥΣ ΤΟΥΣ ΠΙΘΑΝΟΥΣ ΣΥΝΔΙΑΣΜΟΥΣ

ΔΕΝ ΜΠΟΡΕΙ ΝΑ ΛΥΘΕΙ ΜΕ HUNGARIAN - ΤΟΥΛΑΧΙΣΤΟΝ ΔΕΝ ΒΡΗΚΑ FORMULATION ΓΙΑ ΝΑ ΛΥΘΕΙ

Ο Κ ΘΑ ΠΡΕΠΕ ΝΑ ΚΑΝΕΙ ΤΟ ΤΕΛΕΙΟ DET-TRACK ASSOCIATION
ΒΑΖΩΝΤΑΣ ΤΟΝ ΟΜΩΣ ΩΣ D ΒΓΑΖΕΙ ΠΟΛΥ ΚΑΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΕ ΠΟΛΥ ΜΙΚΡΟ RECALL
ΓΙΑ ΚΑΠΟΙΟ ΛΟΓΟ ΚΑΝΕΙ ΣΧΕΤΙΚΑ ΚΑΛΟ ASSOCIATION MONO TO BUS


'''


from scipy.optimize import linear_sum_assignment

def construct_K_matrix8(distance_matrix, dets, curr_gts, trks, prev_gts, threshold=2):

    K = np.ones_like(distance_matrix)

    dets_array = np.array([det[:2] for det in dets])
    curr_gts_array = np.array([gt[:2] for gt in curr_gts], dtype=float)
    trks_array = np.array([trk[:2] for trk in trks])
    prev_gts_array = np.array([gt[:2] for gt in prev_gts], dtype=float)

    curr_gts_ids = np.array([gt[2] for gt in curr_gts])
    prev_gts_ids = np.array([gt[2] for gt in prev_gts])

    # Compute all pairwise distances
    det_gt_distances = np.linalg.norm(dets_array[:, np.newaxis] - curr_gts_array, axis=2)
    trk_gt_distances = np.linalg.norm(trks_array[:, np.newaxis] - prev_gts_array, axis=2)

    # Find close ground truths
    det_gt_close = det_gt_distances <= threshold
    trk_gt_close = trk_gt_distances <= threshold

    total_distance_matrix = np.full((len(dets), len(trks)), np.inf)  # Initialize with infinity

    # Find matching IDs between current and previous ground truths
    for d in range(len(dets)):
        for t in range(len(trks)):
            close_curr_gts = curr_gts_ids[det_gt_close[d]]
            close_prev_gts = prev_gts_ids[trk_gt_close[t]]

            # Find matching IDs
            matching_ids = np.intersect1d(close_curr_gts, close_prev_gts)

            if len(matching_ids) > 0:
                # Store matching information
                for match_id in matching_ids:
                    curr_gt_idx = np.where(curr_gts_ids == match_id)[0][0]
                    prev_gt_idx = np.where(prev_gts_ids == match_id)[0][0]

                    det_distance = det_gt_distances[d, curr_gt_idx]
                    trk_distance = trk_gt_distances[t, prev_gt_idx]

                    total_distance = det_distance + trk_distance
                    total_distance_matrix[d, t] = min(total_distance_matrix[d, t], total_distance)

    print("Total distance matrix:", total_distance_matrix)

    # Use the Hungarian algorithm to find the optimal global assignment
    row_ind, col_ind = linear_sum_assignment(total_distance_matrix)

    # Update the K matrix based on the optimal assignment
    for r, c in zip(row_ind, col_ind):
        if total_distance_matrix[r, c] < np.inf:  # If it's a valid assignment
            K[r, c] = 0  # Set the K matrix entry to 0 for assigned pairs
    print(K)
    return K


def construct_K_matrix7(distance_matrix, dets, curr_gts, trks, prev_gts, threshold=2):
    K = np.ones_like(distance_matrix) 

    dets_array = np.array([det[:2] for det in dets])
    curr_gts_array = np.array([gt[:2] for gt in curr_gts], dtype=float)
    trks_array = np.array([trk[:2] for trk in trks])
    prev_gts_array = np.array([gt[:2] for gt in prev_gts], dtype=float)

    curr_gts_ids = np.array([gt[2] for gt in curr_gts])
    prev_gts_ids = np.array([gt[2] for gt in prev_gts])

    # Compute all pairwise distances
    det_gt_distances = np.linalg.norm(dets_array[:, np.newaxis] - curr_gts_array, axis=2)
    trk_gt_distances = np.linalg.norm(trks_array[:, np.newaxis] - prev_gts_array, axis=2)
    print("det_gt_distances", det_gt_distances, trk_gt_distances)
    # Find close ground truths
    det_gt_close = det_gt_distances <= threshold
    trk_gt_close = trk_gt_distances <= threshold
    print(det_gt_close, trk_gt_close)
    # Dictionary to store matching information
    matching_info = {}

    # Find matching IDs between current and previous ground truths
    for d in range(len(dets)):
        for t in range(len(trks)):
            close_curr_gts = curr_gts_ids[det_gt_close[d]]
            close_prev_gts = prev_gts_ids[trk_gt_close[t]]

            # Find matching IDs
            matching_ids = np.intersect1d(close_curr_gts, close_prev_gts)

            if len(matching_ids) > 0:
                # Store matching information
                for match_id in matching_ids:
                    curr_gt_idx = np.where(curr_gts_ids == match_id)[0][0]
                    prev_gt_idx = np.where(prev_gts_ids == match_id)[0][0]

                    det_distance = det_gt_distances[d, curr_gt_idx]
                    trk_distance = trk_gt_distances[t, prev_gt_idx]

                    if (d, t) not in matching_info:
                        matching_info[(d, t)] = []

                    matching_info[(d, t)].append({
                        'id': match_id,
                        'det_distance': det_distance,
                        'trk_distance': trk_distance,
                        'total_distance': det_distance + trk_distance
                    })
    print(matching_info)

    # To keep track of assigned IDs
    assigned_ids = set()

    # Iterate through matching_info to assign each ID only once
    for (d, t), matches in matching_info.items():

        # Sort matches by total_distance (smallest first)
        matches.sort(key=lambda x: x['total_distance'])

        for match in matches:
            if match['id'] not in assigned_ids:
                # If the ID hasn't been assigned yet, assign it
                K[d, t] = 0  # Set the K matrix entry to 0
                assigned_ids.add(match['id'])  # Mark the ID as assigned
                break  # Break out after assigning one ID for this detection-track pair
    print(K)
    return K



NUSCENES_TRACKING_NAMES = [
    # 'bicycle',
    # 'bus',
    'car',
    # 'motorcycle',
    'pedestrian',
    # 'trailer',
    # 'truck'
]

import pickle

with open('PROB_3D_MULMOD_MOT/scene_data.pkl', 'rb') as f:
    scene_data = pickle.load(f)

for sample_token, sample_data in scene_data.items():
    print(f"Processing sample: {sample_token}")
    for tracking_name in NUSCENES_TRACKING_NAMES:
        dets = sample_data[tracking_name]['dets']
        current_gts = sample_data[tracking_name]['current_gts']
        print(dets, current_gts)






# Plotting
plt.figure(figsize=(8, 8))

# Plot detections
plt.scatter(dets[:, 0], dets[:, 1], color='red', marker='o', label='Detections')

# Plot tracks
plt.scatter(tracks[:, 0], tracks[:, 1], color='blue', marker='x', label='Tracks')

# Plot current ground truth with labels
for gt in curr_gts:
    plt.scatter(float(gt[0]), float(gt[1]), color='green', marker='s', label='Current GT' if gt[2] == curr_gts[0][2] else "")
    plt.text(float(gt[0]) + 0.1, float(gt[1]) + 0.1, str(gt[2]), color='green')

# Plot previous ground truth with labels
for gt in prev_gts:
    plt.scatter(float(gt[0]), float(gt[1]), color='purple', marker='^', label='Previous GT' if gt[2] == prev_gts[0][2] else "")
    plt.text(float(gt[0]) + 0.1, float(gt[1]) + 0.1, str(gt[2]), color='purple')

# Formatting
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Detections, Tracks, and Ground Truths")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('foo.png')

K = construct_K_matrix7(d, dets, curr_gts, tracks, prev_gts)

