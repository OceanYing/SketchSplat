import torch
import math


### ------------- 1. For topo_connect_endpoint() ------------- ###

def connect_endpts(endpts, threshold=0.05):
    N = endpts.shape[0]
    parent = list(range(N))  # Each point is initially its own parent

    def find(i):
        # Path compression for efficiency
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi == pj:
            return
        # Always choose the smaller index as the representative
        if pi < pj:
            parent[pj] = pi
        else:
            parent[pi] = pj

    dists = torch.cdist(endpts, endpts) # Compute pairwise Euclidean distances between points.
    mask = dists < threshold    # Create a boolean mask where distances are below the threshold.

    # Get the index pairs (i, j) where mask is True.
    # torch.nonzero returns all indices; we only consider i < j to avoid duplicates.
    pairs = torch.nonzero(mask, as_tuple=False)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]  # Filter out duplicate and self comparisons

    # Perform union operations for each close pair.
    for i, j in pairs.tolist():
        union(i, j)

    # Group points by their representative.
    groups = {}
    for i in range(N):
        rep = find(i)
        if i != rep:
            groups.setdefault(rep, []).append(i)
    # print('groups:', groups)

    # Only return groups with more than one point.
    merge_groups = {rep: indices for rep, indices in groups.items() if len(indices) > 0}
    # print('merge_groups:', merge_groups)

    return merge_groups



### ------------- 2. For topo_merge_sketch() ------------- ###

import torch

def boxes_close(min_a, max_a, min_b, max_b, threshold):
    """
    Returns True if the two bounding boxes (each defined by its min and max corner)
    overlap along all axes when each box is effectively expanded by 'threshold'.
    """
    for d in range(3):
        if (min_a[d] - threshold > max_b[d]) or (min_b[d] - threshold > max_a[d]):
            return False
    return True

def merge_overlapping_strokes(samp_pts_list, samp_strokeid_list, merge_threshold=0.05, overlap_ratio=0.8):
    """
    Determines which strokes should be deleted because they are almost completely overlapped
    by another stroke. For each stroke pair A and B (with A ≠ B) that are spatially close
    (based on their axis‐aligned bounding boxes), the function computes:
      - frac_A: fraction of A's points that are within merge_threshold of some point in B.
      - frac_B: fraction of B's points that are within merge_threshold of some point in A.
      
    If frac_A ≥ overlap_ratio, then stroke A is marked for deletion.
    Likewise, if frac_B ≥ overlap_ratio, then stroke B is marked for deletion.
    
    Parameters:
      samp_pts_list: torch.Tensor of shape (M, 3)
                     The sampled 3D points from all strokes.
      samp_strokeid_list: 1D tensor or list of length M indicating which stroke each point belongs to.
      merge_threshold: float, distance threshold to consider two sample points as "close."
      overlap_ratio: float (e.g., 0.8)
                     Minimum fraction of points required for a stroke to be considered overlapped.
    
    Returns:
      deletion_list: a sorted list of stroke IDs that should be deleted.
    """
    # Ensure the stroke id list is a tensor.
    if not isinstance(samp_strokeid_list, torch.Tensor):
        samp_strokeid_list = torch.tensor(samp_strokeid_list)
    
    # Get unique stroke IDs.
    unique_stroke_ids = torch.unique(samp_strokeid_list).tolist()
    
    # Efficiently group sample points by stroke id using vectorized indexing.
    stroke_to_pts = {}
    for stroke_id in unique_stroke_ids:
        mask = samp_strokeid_list == stroke_id
        stroke_to_pts[stroke_id] = samp_pts_list[mask]
    
    # Compute the axis-aligned bounding box (AABB) for each stroke.
    stroke_to_bbox = {}
    for stroke_id, pts in stroke_to_pts.items():
        min_box = torch.min(pts, dim=0).values
        max_box = torch.max(pts, dim=0).values
        stroke_to_bbox[stroke_id] = (min_box, max_box)
    
    deletion_set = set()
    n = len(unique_stroke_ids)
    
    # Compare each pair of strokes.
    for i in range(n):
        stroke_A = unique_stroke_ids[i]
        # Skip if already marked for deletion.
        if stroke_A in deletion_set:
            continue
        pts_A = stroke_to_pts[stroke_A]
        min_A, max_A = stroke_to_bbox[stroke_A]
        
        for j in range(i + 1, n):
            stroke_B = unique_stroke_ids[j]
            if stroke_B in deletion_set:
                continue
            if stroke_A == stroke_B:
                continue
            
            pts_B = stroke_to_pts[stroke_B]
            min_B, max_B = stroke_to_bbox[stroke_B]
            
            # Use AABB test to skip strokes that are clearly far apart.
            if not boxes_close(min_A, max_A, min_B, max_B, merge_threshold):
                continue
            
            # Compute pairwise distances between points of stroke A and stroke B.
            dists = torch.cdist(pts_A, pts_B)
            
            # For stroke A: fraction of points that are within merge_threshold of any point in B.
            min_dists_A, _ = torch.min(dists, dim=1)
            frac_A = (min_dists_A < merge_threshold).float().mean().item()
            
            # For stroke B: fraction of points that are within merge_threshold of any point in A.
            min_dists_B, _ = torch.min(dists, dim=0)
            frac_B = (min_dists_B < merge_threshold).float().mean().item()
            
            # If stroke A is mostly covered by stroke B, mark A for deletion.
            if frac_A >= overlap_ratio:
                deletion_set.add(stroke_A)
                # No need to check further for A.
                break
            
            # Also, if stroke B is mostly covered by stroke A, mark B for deletion.
            if frac_B >= overlap_ratio:
                deletion_set.add(stroke_B)
                # Continue checking A against other strokes.
    
    deletion_list = sorted(list(deletion_set))
    return deletion_list


# # Example usage:
# if __name__ == '__main__':
#     # Create three strokes:
#     # Stroke 1 and Stroke 2 are almost identical; Stroke 3 is separate.
#     stroke1 = torch.tensor([[0.0, 0.0, 0.0],
#                             [0.1, 0.0, 0.0],
#                             [0.2, 0.0, 0.0],
#                             [0.3, 0.0, 0.0]])
#     stroke2 = torch.tensor([[0.0, 0.01, 0.0],
#                             [0.1, 0.01, 0.0],
#                             [0.2, 0.01, 0.0],
#                             [0.3, 0.01, 0.0]])
#     stroke3 = torch.tensor([[1.0, 1.0, 1.0],
#                             [1.1, 1.0, 1.0],
#                             [1.2, 1.0, 1.0],
#                             [1.3, 1.0, 1.0]])
    
#     # Combine sample points and create a corresponding stroke id list.
#     samp_pts_list = torch.cat([stroke1, stroke2, stroke3], dim=0)
#     samp_strokeid_list = torch.tensor([1]*4 + [2]*4 + [3]*4)
    
#     strokes_to_delete = merge_overlapping_strokes(samp_pts_list, samp_strokeid_list,
#                                                   merge_threshold=0.05, overlap_ratio=0.8)
#     print("Strokes to delete:", strokes_to_delete)






def merge_colinear_lines(points, idxlist, angle_thresh=5, offset_tol=0.02, overlap_epsilon=0.01):
    """
    Given a set of 2D points (N, 2) and a set of lines (M, 2) where each line is represented
    by a pair of indices into 'points', returns a dict mapping a representative line index to 
    a list of other line indices that are nearly colinear and overlapping with it.
    
    Parameters:
      points: torch.Tensor of shape (N, 2)
              The 2D coordinates.
      idxlist: torch.Tensor of shape (M, 2)
               Each row holds the indices (into 'points') for the endpoints of a line.
      angle_thresh: float
                  Maximum allowed angle (in degrees) between line directions to be considered colinear.
      offset_tol: float
                  Maximum allowed perpendicular distance (offset) from one line's endpoint to the other line.
      overlap_epsilon: float
                  Minimum required overlap length (in projected units) for the segments.
    
    Returns:
      merge_dict: dict
          Keys are representative line indices (from 0 to M-1) and the values are sorted lists 
          of the other line indices that should be merged with the key.
    """
    idxorig = idxlist.clone()
    M = idxlist.shape[0]
    # Initialize union-find structure.
    parent = list(range(M))
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    
    def union(i, j, new_line):
        pi = find(i)
        pj = find(j)
        if pi != pj:
            # Choose representative by the smaller index.
            if pi < pj:
                parent[pj] = pi
                # Update the representative's idxlist value.
            else:
                parent[pi] = pj
            idxlist[pi, 0] = new_line[0]
            idxlist[pi, 1] = new_line[1]
            idxlist[pj, 0] = new_line[0]
            idxlist[pj, 1] = new_line[1]
    
    def get_offset(p_i, d_i_n, p_j):
        diff = p_j - p_i
        proj_diff = torch.dot(diff, d_i_n)
        closest = p_i + proj_diff * d_i_n
        offset = torch.norm(p_j - closest).item()
        return offset
    
    # Precompute cosine threshold (angle_thresh is given in degrees)
    cos_thresh = math.cos(math.radians(angle_thresh))
    
    # Loop over each pair of lines.
    for i in range(M):
        # Endpoints for line i:

        i_parent = find(i)
        idxlist[i, :] = idxlist[i_parent, :]    # update current line as union line

        p_i1 = points[idxlist[i, 0]]
        p_i2 = points[idxlist[i, 1]]
        d_i = p_i2 - p_i1
        norm_i = torch.norm(d_i)
        if norm_i < 1e-8:
            continue  # skip degenerate line
        d_i_n = d_i / norm_i
        
        # Compute the 1D projection interval for line i on direction d_i_n.
        proj_i1 = torch.dot(p_i1, d_i_n)
        proj_i2 = torch.dot(p_i2, d_i_n)
        interval_i = (min(proj_i1.item(), proj_i2.item()),
                      max(proj_i1.item(), proj_i2.item()))
        
        for j in range(i+1, M):
            j_parent = find(j)
            idxlist[j, :] = idxlist[j_parent, :]    # update current line as union line
            # Endpoints for line j:
            p_j1 = points[idxlist[j, 0]]
            p_j2 = points[idxlist[j, 1]]
            d_j = p_j2 - p_j1
            norm_j = torch.norm(d_j)
            if norm_j < 1e-8:
                continue
            d_j_n = d_j / norm_j
            
            # Check if the lines are nearly colinear by comparing their direction vectors.
            dot_val = torch.dot(d_i_n, d_j_n).item()
            if abs(dot_val) < cos_thresh:
                continue
            
            # Check the offset: compute distance from p_j1 to the infinite line defined by line i.
            offset_1 = get_offset(p_i1, d_i_n, p_j1)
            offset_2 = get_offset(p_i1, d_i_n, p_j2)
            
            if max(offset_1, offset_2) > offset_tol:
                continue
            
            # Project endpoints of line j onto d_i_n.
            proj_j1 = torch.dot(p_j1, d_i_n)
            proj_j2 = torch.dot(p_j2, d_i_n)
            interval_j = (min(proj_j1.item(), proj_j2.item()),
                          max(proj_j1.item(), proj_j2.item()))
            
            # Check overlap between the intervals.
            overlap_start = max(interval_i[0], interval_j[0])
            overlap_end = min(interval_i[1], interval_j[1])


            if overlap_end > overlap_start - overlap_epsilon:

                proj_list = torch.tensor([proj_i1.item(), proj_i2.item(), proj_j1.item(), proj_j2.item()])
                pidx_list = torch.tensor([idxlist[i, 0].item(), idxlist[i, 1].item(), idxlist[j, 0].item(), idxlist[j, 1].item()])
                max_pos = pidx_list[torch.argmax(proj_list)]
                min_pos = pidx_list[torch.argmin(proj_list)]
                new_line = (min_pos, max_pos)

                # print(pidx_list.tolist(), " -> ", new_line)
                # print(idxlist)

                # ll = [25,3,5,16,52,7,58,10]
                # if (i in ll) or (j in ll):
                #     # import pdb
                #     # pdb.set_trace()
                #     print("merge:", i, j, idxlist[i], idxlist[j], find(i), find(j), idxlist[find(i)], idxlist[find(j)])
                #     print("orig:", idxorig[i], idxorig[j], points[idxorig[i]], points[idxorig[j]])
                #     print()

                
                union(i, j, new_line)

                idxlist[i, :] = idxlist[find(i), :]    # update current line as union line
                idxlist[j, :] = idxlist[find(j), :]    # update current line as union line

                # print(idxlist)
    
    # Group line indices by their representative.
    groups = {}
    for i in range(M):
        rep = find(i)
        groups.setdefault(rep, []).append(i)
    
    # Only return groups that contain more than one line.
    merge_dict = {rep: sorted([idx for idx in group if idx != rep])
                  for rep, group in groups.items() if len(group) > 1}
    # merge_dict = {rep: sorted([idx for idx in group])
    #               for rep, group in groups.items() if len(group) > 1}
    

    for i in range(M):      # update all line's point idx as parent line
        i_parent = find(i)
        idxlist[i, :] = idxlist[i_parent, :]

    return merge_dict, idxlist


# # Example usage:
# if __name__ == '__main__':
#     # Define some 2D points.
#     pts = torch.tensor([
#         [0.0, 0.0],
#         [1.0, 0.0],
#         [2.0, 0.0],
#         [3.0, 0.0],
#         [0.0, 0.1],
#         [1.0, 0.1],
#         [2.0, 0.1],
#         [3.0, 0.1],
#         [0.0, 1.0],
#         [1.0, 1.0],

#         [0.0, 0.0],
#         [1.0, 0.0],
#         [2.0, 0.0],
#         [3.0, 0.0],
#         [0.1, 0.0],
#         [1.1, 0.0],
#         [0.5, 0.0],
#         [2.5, 0.0],
#     ])
    
#     # Define lines via an index list.
#     # For example, line 0: from pts[0] to pts[1], line 1: pts[1] to pts[2], etc.
#     idxlist = torch.tensor([
#         # [0, 1],
#         # [1, 2],
#         # [2, 3],
#         # [4, 5],
#         # [5, 6],
#         # [6, 7],
#         # [8, 9]
#         [10, 11],
#         [12, 13],
#         [14, 15],
#         # [16, 17]
#     ])
    
#     # Merge nearly colinear and overlapping lines.
#     merge_dict = merge_lines(pts, idxlist, angle_thresh=10, offset_tol=0.15, overlap_epsilon=1e-6)
#     print("Lines to merge:", merge_dict)