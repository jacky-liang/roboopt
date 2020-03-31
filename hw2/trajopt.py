import os
import argparse

import numpy as np
import torch
from torch.autograd import grad
from torch_utils import from_numpy, get_numpy, ones, tensor, zeros_like

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from constants import CS


def cubic_hermite(ts, x0, x1, v0, v1):
    '''
    From https://www.rose-hulman.edu/~finn/CCLI/Notes/day09.pdf
    '''
    t2 = ts * ts
    t3 = t2 * ts
    
    H0 = 1 - 3*t2 + 2*t3
    H1 = ts - 2*t2 + t3
    H2 = -t2 + t3
    H3 = 3*t2 - 2*t3
    
    return H0.reshape(-1, 1) * x0 + H1.reshape(-1, 1) * v0 + H2.reshape(-1, 1) * v1 + H3.reshape(-1, 1) * x1


def cubic_hermite_d(ts, x0, x1, v0, v1):
    t2 = ts * ts
    
    H0 = - 6*ts + 6*t2
    H1 = ones(len(ts)) - 4*ts + 3*t2
    H2 = -2*ts + 3*t2
    H3 = 6*ts - 6*t2
    
    return H0.reshape(-1, 1) * x0 + H1.reshape(-1, 1) * v0 + H2.reshape(-1, 1) * v1 + H3.reshape(-1, 1) * x1


def gen_trajs(wps, wp_speeds, n_pts):
    trajs = []
    d_trajs = []

    ts = from_numpy(np.linspace(0, 1, n_pts))
    
    vs = torch.stack([torch.cos(wps[:,2]), torch.sin(wps[:,2])]).transpose(0, 1)
    vs[0] *= 1e-2
    vs[1:-1] *= torch.unsqueeze(wp_speeds, 1)
    vs[-1] *= 1e-2
    
    for i in range(1, len(wps)):
        cur_x, cur_v = wps[i - 1, :2], vs[i - 1]
        next_x, next_v = wps[i, :2], vs[i]

        traj = cubic_hermite(ts, cur_x, next_x, cur_v, next_v)
        trajs.append(traj)

        d_traj = cubic_hermite_d(ts, cur_x, next_x, cur_v, next_v)
        d_trajs.append(d_traj)

    trajs = torch.stack(trajs)
    d_trajs = torch.stack(d_trajs)
    
    return trajs, d_trajs


def trajopt(wps, writer, n_pts=40, constraint_weights=[0.01, 1, 0.0001, 0.001, 0.1, 0.1], max_n_opts=1000):
    # define params
    constraint_weights = from_numpy(np.array(constraint_weights))
    wps_trch = tensor(wps, requires_grad=True)
    wp_speeds = tensor(np.ones(len(wps) - 2) * 30, requires_grad=True)
    seg_times = tensor(np.ones(len(wps) - 1) * 10, requires_grad=True)

    # define bounds
    wps_delta = from_numpy(np.array([CS['waypoint_tol'], CS['waypoint_tol'], np.deg2rad(CS['angle_tol'])]))
    wps_lo, wps_hi = wps_trch - wps_delta, wps_trch + wps_delta

    seg_times_lo = from_numpy(np.linalg.norm(wps[1:, :2] - wps[:-1, :2], axis=1) / CS['max_vel'])

    seg_speed_lo, seg_speed_hi = from_numpy(np.array(0)), from_numpy(np.array(CS['max_vel']))

    trajs_lo, trajs_hi = from_numpy(CS['xyp_lims_lo']), from_numpy(CS['xyp_lims_hi'])

    n_opts = trange(max_n_opts)
    for n_opt in n_opts:
        # compute trajs
        trajs, d_trajs = gen_trajs(wps_trch, wp_speeds, n_pts)

        # compute needed values
        velocities = d_trajs / torch.unsqueeze(torch.unsqueeze(seg_times, 1), 1)

        speeds = torch.norm(velocities, dim=2)
        accelerations = (speeds[:, 1:] - speeds[:, :-1]) / torch.unsqueeze(seg_times, 1) * n_pts

        angles = torch.atan2(trajs[:,:,1], trajs[:,:,0])
        betas = torch.asin(torch.tanh(CS['wheelbase'] / 2 * angles / speeds))
        steering_angles = torch.atan(2 * torch.tan(betas))

        velocities_pred = torch.unsqueeze(speeds, 2) * torch.cat([
                                                                torch.unsqueeze(torch.cos(angles + betas), 2), 
                                                                torch.unsqueeze(torch.sin(angles + betas), 2)
                                                                ], 2)

        # compute loss
        total_time = torch.sum(seg_times)

        constraint_costs = torch.stack([
            # dynamics
            torch.norm(velocities_pred - velocities),
            # steering angle
            torch.sum(torch.relu(torch.pow(steering_angles, 2) - CS['max_steering_angle']**2)),
            # acceleration
            torch.sum(torch.relu(torch.pow(accelerations, 2) - CS['max_acc']**2)),
            # speed
            torch.sum(torch.relu(torch.pow(speeds, 2) - CS['max_vel']**2)),
            # traj bounds
            torch.norm(torch.relu(trajs - trajs_hi)),
            torch.norm(torch.relu(-trajs + trajs_lo))    
        ])

        constraint_cost = constraint_costs @ constraint_weights
        loss = total_time + constraint_cost

        # TODO: update params by grad

        # TODO: project onto bounds

    return {
        'wps_trch': wps_trch,
        'wp_speeds': wp_speeds,
        'seg_times': seg_times,
        'trajs': trajs,
        'd_trajs': d_trajs,
        'velocities': velocities,
        'speeds': speeds,
        'accelerations': accelerations,
        'steering_angles': steering_angles,
        'velocities_pred': velocities_pred,
        'betas': betas,
        'total_time': total_time,
        'loss': loss,
        'constraint_costs': constraint_costs,
        'constraint_cost': constraint_cost
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', '-l', type=str, default='outs')
    parser.add_argument('--tag', type=str, required=True)
    args = parser.parse_args()

    wps = np.array([
            [0, 0, 0],
            [ 8.62844369,  4.89809566,  1.53850753],
            [-3.3128731 , -8.41245037,  0.2807953 ],
            [ 6.42118345,  9.56369873, -0.52140243],
            [-3.55531364, -7.33024669, -0.58341082],
            [-4.72198946, -4.64057374, -1.95494644],
            [-1.21513531, -4.07460913, -1.61876955],
            [-4.51332572,  9.87373577,  0.22790191],
            [-6.26747526, -9.75024744, -1.55346495],
            [ 7.99872313, -0.73949926, -2.41688927],
            [-9.2515838 ,  6.51287078, -0.61068366]
    ])

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, args.tag))
    res = trajopt(wps, writer)

    import IPython; IPython.embed(); exit(0)
