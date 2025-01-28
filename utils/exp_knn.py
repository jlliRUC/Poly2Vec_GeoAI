import numpy as np
import torch
from tqdm import tqdm


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def point_to_segment_distance_m(point, segment_start, segment_end):
    px, py = point
    sx1, sy1 = segment_start
    sx2, sy2 = segment_end

    dx = sx2 - sx1
    dy = sy2 - sy1
    segment_length_squared = dx * dx + dy * dy

    # If the segment's length is zero (both endpoints are the same)
    if segment_length_squared == 0:
        return euclidean_distance(point, segment_start)

    # Compute the projection of the point onto the line segment
    t = ((px - sx1) * dx + (py - sy1) * dy) / segment_length_squared

    # Determine the closest point on the segment
    if t < 0:
        closest_point = (sx1, sy1)  # Point is beyond the start of the segment
    elif t > 1:
        closest_point = (sx2, sy2)  # Point is beyond the end of the segment
    else:
        closest_point = (sx1 + t * dx, sy1 + t * dy)  # Point is within the segment

    # Return the Euclidean distance from the point to the closest point
    return euclidean_distance(point, closest_point)


def compute_min_distance(point, polyline):
    min_dist = float('inf')

    for i in range(len(polyline) - 1):
        segment_start = polyline[i]
        segment_end = polyline[i + 1]

        # Distance to both endpoints of the segment
        dist_to_start = euclidean_distance(point, segment_start)
        dist_to_end = euclidean_distance(point, segment_end)

        # Perpendicular distance to the segment
        dist_perpendicular = point_to_segment_distance_m(point, segment_start, segment_end)

        # Take the minimum of the distances
        min_dist = min(min_dist, dist_to_start, dist_to_end, dist_perpendicular)

    return min_dist


# def point_to_segment_distance(point, segment_start, segment_end, distance_metric):
#     # Vector from segment_start to segment_end
#     segment_vector = segment_end - segment_start
#     # Vector from segment_start to the point
#     point_vector = point - segment_start

#     # Compute the projection of the point onto the line segment
#     segment_length_squared = np.dot(segment_vector, segment_vector)
#     if segment_length_squared == 0:
#         # segment_start and segment_end are the same point
#         return np.linalg.norm(point_vector)

#     t = np.dot(point_vector, segment_vector) / segment_length_squared
#     if t < 0.0:
#         projection = segment_start
#     elif t > 1.0:
#         projection = segment_end
#     else:
#         projection = segment_start + t * segment_vector

#     # Compute the distance from the point to the projection
#     distance = distance_metric(point - projection)
#     return distance

# def point_to_polyline_distance(point, polyline, distance_metric):
#     # polyline is a sequence of points (N x 2 array)
#     min_distance = float('inf')
#     for i in range(len(polyline) - 1):
#         segment_start = polyline[i]
#         segment_end = polyline[i + 1]
#         distance = point_to_segment_distance(point, segment_start, segment_end, distance_metric)
#         if distance < min_distance:
#             min_distance = distance
#     return min_distance


def point_to_geometry_distance(point, object, object_type, distance_metric):
    if object_type == "point":
        return distance_metric(point - object)
    elif object_type == "polyline":
        # TODO
        distance = compute_min_distance(point, object)
        return distance
        # return point_to_polyline_distance(point, object, distance_metric)

    elif object_type == "polygon":
        cendroid = np.mean(object, axis=0)
        return distance_metric(point - cendroid)


def generate_ground_truth_distance(subjects, objects, object_type, distance_metric=np.linalg.norm):
    dis_matrix = []
    for subject in tqdm(subjects):
        dis_temp = []
        for object in objects:
            dis_temp.append(point_to_geometry_distance(subject.clone().detach().cpu().numpy(),
                                                       object.clone().detach().cpu().numpy(), object_type,
                                                       distance_metric))
        dis_matrix.append(dis_temp)

    return np.array(dis_matrix)


def generate_pairwise_ground_truth_distance(subjects, objects, object_type, distance_metric=np.linalg.norm):
    dis_matrix = []
    for i in range(len(subjects)):
        dis_matrix.append(point_to_geometry_distance(subjects[i].clone().detach().cpu().numpy(),
                                                     objects[i].clone().detach().cpu().numpy(),
                                                     object_type,
                                                     distance_metric))

    return torch.tensor(dis_matrix, dtype=torch.float32)


class KNNRegressionLoss(torch.nn.Module):
    def __init__(self):
        super(KNNRegressionLoss, self).__init__()

    def forward(self, subject_embeddings, object_embeddings, ground_truth):
        pred_list = []
        for object_emb in object_embeddings:
            pred = torch.norm(subject_embeddings - object_emb, dim=1)
            pred_list.append(pred.unsqueeze(-1))
        pred = torch.concat(pred_list, dim=1)

        loss = torch.nn.functional.mse_loss(pred, ground_truth)
        return loss


def knn(subject_embeddings, object_embeddings, ground_truth_matrix, k1, k2, k3, t3):
    subject_embeddings = subject_embeddings.data.cpu().numpy()
    object_embeddings = object_embeddings.data.cpu().numpy()
    print(subject_embeddings.shape)

    pred_dis_matrix = []
    for t in subject_embeddings:
        emb = np.repeat([t], repeats=len(subject_embeddings), axis=0)
        matrix = np.linalg.norm(emb - object_embeddings, ord=2, axis=1)
        pred_dis_matrix.append(matrix.tolist())

    HR_k1_temp = 0
    HR_k2_temp = 0
    R_k3_temp = 0

    f_num = len(ground_truth_matrix)
    for i in range(f_num):
        ground_truth = np.array(ground_truth_matrix[i])
        true_sorted = np.argsort(ground_truth)
        true_k1 = true_sorted[:k1]
        true_k2 = true_sorted[:k2]
        true_k3 = true_sorted[:k3]

        pred = np.array(pred_dis_matrix[i])
        pred_sorted = np.argsort(pred)
        pred_k1 = pred_sorted[:k1]
        pred_k2 = pred_sorted[:k2]
        pred_t3 = pred_sorted[:t3]

        HR_k1_temp += len(list(set(true_k1).intersection(set(pred_k1))))
        HR_k2_temp += len(list(set(true_k2).intersection(set(pred_k2))))
        R_k3_temp += len(list(set(true_k3).intersection(set(pred_t3))))

    HR_k1 = float(HR_k1_temp) / (k1 * f_num)
    HR_k2 = float(HR_k2_temp) / (k2 * f_num)
    R_k3 = float(R_k3_temp) / (k3 * f_num)

    return HR_k1, HR_k2, R_k3
