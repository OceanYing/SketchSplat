import torch
import torch.jit
# from sketchgaussians.utils.eval_utils import bezier_length_simpson_fast
from sketchgaussians.utils.eval_utils import bezier_length_simpson_fast_torch

def compute_curve_points(curve, sample_resolution):
    """ Compute sampled points and directions for a single cubic Bezier curve. """
    # each_curve_np = curve.detach().cpu().numpy()
    # sample_num = int(
    #     bezier_length_simpson_fast_torch(each_curve_np[0, :],
    #                                each_curve_np[1, :],
    #                                each_curve_np[2, :],
    #                                each_curve_np[3, :],
    #                                n=100) // sample_resolution
    # )

    each_curve = curve
    sample_num = bezier_length_simpson_fast_torch(each_curve[0, :],
                                   each_curve[1, :],
                                   each_curve[2, :],
                                   each_curve[3, :],
                                   n=100).int()

    t = torch.linspace(0, 1, sample_num, device=curve.device)
    matrix_u = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=0)  # (4, sample_num)

    matrix_middle = torch.Tensor(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
    ).to(curve.device)

    points = torch.matmul(torch.matmul(matrix_u.T, matrix_middle), curve).reshape(sample_num, 3)

    # Compute curve directions (first derivative)
    derivative_u = 3 * t**2
    derivative_v = 2 * t

    dx = (
        (-3 * curve[0][0] + 9 * curve[1][0] - 9 * curve[2][0] + 3 * curve[3][0]) * derivative_u
        + (6 * curve[0][0] - 12 * curve[1][0] + 6 * curve[2][0]) * derivative_v
        + (-3 * curve[0][0] + 3 * curve[1][0])
    )

    dy = (
        (-3 * curve[0][1] + 9 * curve[1][1] - 9 * curve[2][1] + 3 * curve[3][1]) * derivative_u
        + (6 * curve[0][1] - 12 * curve[1][1] + 6 * curve[2][1]) * derivative_v
        + (-3 * curve[0][1] + 3 * curve[1][1])
    )

    dz = (
        (-3 * curve[0][2] + 9 * curve[1][2] - 9 * curve[2][2] + 3 * curve[3][2]) * derivative_u
        + (6 * curve[0][2] - 12 * curve[1][2] + 6 * curve[2][2]) * derivative_v
        + (-3 * curve[0][2] + 3 * curve[1][2])
    )

    direction = torch.stack([dx, dy, dz], dim=-1)
    norm_direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)

    return points, norm_direction, torch.ones(sample_num, device=curve.device)


def compute_line_points(line, sample_resolution):
    """ Compute sampled points and directions for a single line segment. """
    sample_num = int(torch.norm(line[0] - line[-1]) // sample_resolution)
    t = torch.linspace(0, 1, sample_num, device=line.device)

    matrix_u_l = torch.stack([t, torch.ones_like(t)], dim=0)
    matrix_middle_l = torch.Tensor([[-1, 1], [1, 0]]).to(line.device)

    points = torch.matmul(torch.matmul(matrix_u_l.T, matrix_middle_l), line).reshape(sample_num, 3)

    # Compute direction
    direction = line[1] - line[0]
    norm_direction = direction / (torch.norm(direction) + 1e-6)

    return points, torch.ones_like(points) * norm_direction, torch.ones(sample_num, device=line.device)


def sample_points_from_curves_diff2(curves_ctl_pts, lines_end_pts, sample_resolution=0.005):
    num_curves = len(curves_ctl_pts)
    num_lines = len(lines_end_pts)

    # Parallel processing for curves
    curve_futures = []
    for i in range(num_curves):
        future = torch.jit.fork(compute_curve_points, curves_ctl_pts[i, :, :].reshape(4, 3), sample_resolution)
        curve_futures.append(future)

    # Parallel processing for lines
    line_futures = []
    for i in range(num_lines):
        future = torch.jit.fork(compute_line_points, lines_end_pts[i, :, :].reshape(2, 3), sample_resolution)
        line_futures.append(future)

    # Collect results from parallel computations
    all_curve_points, all_curve_directions, all_curve_stroke_idx = [], [], []
    for i, future in enumerate(curve_futures):
        points, directions, stroke_idx = torch.jit.wait(future)
        all_curve_points.append(points)
        all_curve_directions.append(directions)
        all_curve_stroke_idx.append(stroke_idx * i)

    all_line_points, all_line_directions, all_line_stroke_idx = [], [], []
    for i, future in enumerate(line_futures):
        points, directions, stroke_idx = torch.jit.wait(future)
        all_line_points.append(points)
        all_line_directions.append(directions)
        all_line_stroke_idx.append(stroke_idx * (i + num_curves))

    # Concatenating results
    if all_curve_points:
        all_curve_points = torch.cat(all_curve_points, dim=0).reshape(-1, 3)
        all_curve_directions = torch.cat(all_curve_directions, dim=0).reshape(-1, 3)
        all_curve_stroke_idx = torch.cat(all_curve_stroke_idx).long()
    else:
        all_curve_points, all_curve_directions, all_curve_stroke_idx = None, None, None

    if all_line_points:
        all_line_points = torch.cat(all_line_points, dim=0).reshape(-1, 3)
        all_line_directions = torch.cat(all_line_directions, dim=0).reshape(-1, 3)
        all_line_stroke_idx = torch.cat(all_line_stroke_idx).long()
    else:
        all_line_points, all_line_directions, all_line_stroke_idx = None, None, None
    
    all_stroke_idx = torch.cat([all_curve_stroke_idx, all_line_stroke_idx]).long()

    return all_curve_points, all_line_points, all_curve_directions, all_line_directions, all_stroke_idx
