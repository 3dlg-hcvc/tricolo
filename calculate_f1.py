import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
import jsonlines
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
import numpy as np
import random
import os
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


import json
with open("data/text2shape-data/shapenet/preprocessed/exp_data/val_map.json", "r") as f:
    data = json.load(f)
amap = {}
for item in data:
    amap[item["model_id"]] = item["category"]

def _compute_sampling_metrics(pred_points, gt_points, thresholds=None, eps=1e-8):
    """
    Compute metrics that are based on sampling points and normals:
    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        # metrics["Precision@%f" % t] = precision
        # metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    # metrics = {k: v.cpu() for k, v in metrics.items()}
    return f1


def _scale_meshes(pred_meshes, gt_meshes, scale):
    if isinstance(scale, float):
        # Assume scale is a single scalar to use for both preds and GT
        pred_scale = gt_scale = scale
    elif isinstance(scale, tuple):
        # Rescale preds and GT with different scalars
        pred_scale, gt_scale = scale
    elif scale.startswith("gt-"):
        # Rescale both preds and GT so that the largest edge length of each GT
        # mesh is target
        target = float(scale[3:])
        bbox = gt_meshes.get_bounding_boxes()  # (N, 3, 2)
        long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
        scale = target / long_edge
        if scale.numel() == 1:
            scale = scale.expand(len(pred_meshes))
        pred_scale, gt_scale = scale, scale
    else:
        raise ValueError("Invalid scale: %r" % scale)
    pred_meshes = pred_meshes.scale_verts(pred_scale)
    gt_meshes = gt_meshes.scale_verts(gt_scale)
    return pred_meshes, gt_meshes


def _sample_meshes(meshes, num_samples):
    """
    Helper to either sample points uniformly from the surface of a mesh
    (with normals), or take the verts of the mesh as samples.
    Inputs:
        - meshes: A MeshList
        - num_samples: An integer, or the string 'verts'
    Outputs:
        - verts: Either a Tensor of shape (N, S, 3) if we take the same number of
          samples from each mesh; otherwise a list of length N, whose ith element
          is a Tensor of shape (S_i, 3)
        - normals: Either a Tensor of shape (N, S, 3) or None if we take verts
          as samples.
    """
    if num_samples == "verts":
        normals = None
        # if meshes.equisized:
        #    verts = meshes.verts_batch
        # else:
        verts = meshes.verts_list()

    else:
        verts, _ = sample_points_from_meshes(meshes, num_samples, return_normals=True)
    return verts, None

def compare_meshes(
        pred_meshes, pred_mesh_name, gt_meshes, gt_mesh_name, num_samples=10000, scale="gt-10", thresholds=None, reduce=False, eps=1e-8
):
    """
    Compute evaluation metrics to compare meshes. We currently compute the
    following metrics:
    - L2 Chamfer distance
    - Normal consistency
    - Absolute normal consistency
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    Inputs:
        - pred_meshes (Meshes): Contains N predicted meshes
        - gt_meshes (Meshes): Contains 1 or N ground-truth meshes. If gt_meshes
          contains 1 mesh, it is replicated N times.
        - num_samples: The number of samples to take on the surface of each mesh.
          This can be one of the following:
            - (int): Take that many uniform samples from the surface of the mesh
            - 'verts': Use the vertex positions as samples for each mesh
            - A tuple of length 2: To use different sampling strategies for the
              predicted and ground-truth meshes (respectively).
        - scale: How to scale the predicted and ground-truth meshes before comparing.
          This can be one of the following:
            - (float): Multiply the vertex positions of both meshes by this value
            - A tuple of two floats: Multiply the vertex positions of the predicted
              and ground-truth meshes by these two different values
            - A string of the form 'gt-[SCALE]', where [SCALE] is a float literal.
              In this case, each (predicted, ground-truth) pair is scaled differently,
              so that bounding box of the (rescaled) ground-truth mesh has longest
              edge length [SCALE].
        - thresholds: The distance thresholds to use when computing precision, recall,
          and F1 scores.
        - reduce: If True, then return the average of each metric over the batch;
          otherwise return the value of each metric between each predicted and
          ground-truth mesh.
        - eps: Small constant for numeric stability when computing F1 scores.
    Returns:
        - metrics: A dictionary mapping metric names to their values. If reduce is
          True then the values are the average value of the metric over the batch;
          otherwise the values are Tensors of shape (N,).
    """
    if thresholds is None:
        thresholds = [0.1, ]  # , 0.3, 0.5] # [0.1, 0.2, 0.3, 0.4 0.5] before
    if not os.path.exists(f"point_cache/{pred_mesh_name}.npy") or not os.path.exists(f"point_cache/{gt_mesh_name}.npy"):
        pred_meshes, gt_meshes = _scale_meshes(pred_meshes, gt_meshes, scale)

    if isinstance(num_samples, tuple):
        num_samples_pred, num_samples_gt = num_samples
    else:
        num_samples_pred = num_samples_gt = num_samples

    # num_samples_pred = num_samples_gt = 'verts'

    ###### sample_meshes Method 1 #####
    pred_points = []
    for pred_mesh in pred_meshes:
        if os.path.exists(f"point_cache/{pred_mesh_name}.npy"):
            pred_point = torch.from_numpy(np.load(f"point_cache/{pred_mesh_name}.npy"))
        else:
            pred_point, _ = _sample_meshes(pred_mesh, num_samples_pred)
            os.makedirs("point_cache", exist_ok=True)
            np.save(f"point_cache/{pred_mesh_name}.npy", pred_point.numpy())
        pred_points.append(pred_point)

    # convert to tensor
    pred_points = torch.concat(pred_points)

    ###### sample_meshes Method 2 #####
    # pred_points, pred_normals = _sample_meshes(pred_meshes, num_samples_pred)

    if os.path.exists(f"point_cache/{gt_mesh_name}.npy"):
        gt_points = torch.from_numpy(np.load(f"point_cache/{gt_mesh_name}.npy"))
    else:
        gt_points, _ = _sample_meshes(gt_meshes, num_samples_gt)
        os.makedirs("point_cache", exist_ok=True)
        np.save(f"point_cache/{gt_mesh_name}.npy", gt_points.numpy())


    if torch.is_tensor(pred_points) and torch.is_tensor(gt_points):

        gt_points = gt_points.expand(len(pred_meshes), -1, -1)
        # We can compute all metrics at once in this case
        f1 = _compute_sampling_metrics(
            pred_points, gt_points, thresholds=thresholds
        )
    else:
        raise NotImplementedError

    return f1

def run_parallel(result):
    # cat_id = result["cat_id"]
    gt_id = result["groundtruth"].split("-")[0]
    pred_ids = result["retrieved_models"]

    if not os.path.exists(f"point_cache/{gt_id}.npy") or not os.path.exists(f"point_cache/{pred_ids[0]}.npy"):
        gt_verts, gt_faces, _ = load_obj(
            os.path.join("data/text2shape-data/ShapeNetCore.v2", amap[gt_id], gt_id, "models", "model_normalized.obj"),
            load_textures=False)  # verts, faces, aux
        gt_mesh = Meshes(verts=[gt_verts], faces=[gt_faces.verts_idx])

        pred_meshes_list = []
        # for pred_id in pred_ids:

        mesh = load_obj(os.path.join("data/text2shape-data/ShapeNetCore.v2", amap[pred_ids[0]], pred_ids[0], "models",
                                     "model_normalized.obj"), load_textures=False)  # verts, faces, aux
        pred_meshes_list.append([mesh[0], mesh[1].verts_idx])


        verts = [mesh[0] for mesh in pred_meshes_list]
        faces = [mesh[1] for mesh in pred_meshes_list]
        pred_meshes = Meshes(verts=verts, faces=faces)
    else:
        pred_meshes = [None]
        gt_mesh = None
    metrics = compare_meshes(pred_meshes, pred_ids[0], gt_mesh, gt_id)
    return metrics.mean().item()


if __name__ == '__main__':
    with jsonlines.open("nearest.jsonl") as reader:
        results = list(reader)

    new_results = []
    for result in results:
        gt_id = result["groundtruth"].split("-")[0]
        pred_ids = result["retrieved_models"]
        if gt_id not in amap:
            continue
        if not os.path.exists(os.path.join("data/text2shape-data/ShapeNetCore.v2", amap[gt_id], gt_id, "models", "model_normalized.obj")):
            continue

        new_results.append(result)

    output = process_map(
        run_parallel, new_results, chunksize=1, max_workers=10
    )

    print(sum(output) / len(output))
    # for result in tqdm(results):
    #     run_parallel(result)
