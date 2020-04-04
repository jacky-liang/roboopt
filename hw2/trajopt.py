import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import torch
from torch.autograd import grad
from torch_utils import from_numpy, get_numpy, ones, tensor, zeros_like

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from constants import CS
from adam import Adam


def plot_trajs(wps, trajs, d_trajs, title='', save=None, show=False):
    plt.figure(figsize=(15, 15))
    ax = plt.gca()

    for i, traj in enumerate(trajs):
        plt.scatter(traj[:,0], traj[:,1])

        for j, pt in enumerate(traj):
            d_pt = d_trajs[i, j] * 0.05
            ax.arrow(pt[0], pt[1], d_pt[0], d_pt[1], head_width=0.2, head_length=0.2)

    arrow_length = 0.8

    plt.scatter(wps[:, 0], wps[:, 1])
    for i, wp in enumerate(wps):
        tip = arrow_length * np.array([np.cos(wp[2]), np.sin(wp[2])])
        ax.arrow(wp[0], wp[1], tip[0], tip[1], head_width=0.3, head_length=0.3)

        ax.text(wp[0] * 1.03, wp[1] * 1.03, '{}'.format(i), fontsize=15)

    lines = [
        [(CS['x_lim'][0], CS['y_lim'][0]), (CS['x_lim'][0], CS['y_lim'][1])], 
        [(CS['x_lim'][0], CS['y_lim'][0]), (CS['x_lim'][1], CS['y_lim'][0])], 
        [(CS['x_lim'][1], CS['y_lim'][1]), (CS['x_lim'][0], CS['y_lim'][1])], 
        [(CS['x_lim'][1], CS['y_lim'][1]), (CS['x_lim'][1], CS['y_lim'][0])], 
        [(CS['xp_lim'][0], CS['yp_lim'][0]), (CS['xp_lim'][0], CS['yp_lim'][1])], 
        [(CS['xp_lim'][0], CS['yp_lim'][0]), (CS['xp_lim'][1], CS['yp_lim'][0])], 
        [(CS['xp_lim'][1], CS['yp_lim'][1]), (CS['xp_lim'][0], CS['yp_lim'][1])], 
        [(CS['xp_lim'][1], CS['yp_lim'][1]), (CS['xp_lim'][1], CS['yp_lim'][0])], 
    ]
    lc = mc.LineCollection(lines, linewidths=2)
    ax.add_collection(lc)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(CS['xp_lim'] * 1.1)
    plt.ylim(CS['yp_lim'] * 1.1)
    plt.title(title)

    ax.set_aspect('equal')
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()


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


def gen_trajs(wps, wp_speeds, init_vel, n_pts):
    trajs = []
    d_trajs = []

    ts = from_numpy(np.linspace(0, 1, n_pts))
    
    vs = torch.stack([torch.cos(wps[:,2]), torch.sin(wps[:,2])]).transpose(0, 1)
    vs[0] = init_vel
    vs[1:-1] *= torch.unsqueeze(wp_speeds, 1)
    vs[-1] *= 1e-2
    
    # TODO: batch this
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


def trajopt(wps, writer, init_vel=[0, 0.01], n_pts=40, constraint_weights=[0.01, 1, 0.0001, 0.001, 0.1, 0.1], max_n_opts=1000, lr=1e-6):
    # define params
    init_vel = from_numpy(np.array(init_vel))
    constraint_weights = from_numpy(np.array(constraint_weights))
    wps_init = tensor(wps, requires_grad=True)
    wp_speeds_init = tensor(np.ones(len(wps) - 2) * 30, requires_grad=True)
    seg_times_init = tensor(np.ones(len(wps) - 1) * 30, requires_grad=True)

    # define bounds
    wps_delta = from_numpy(np.array([CS['waypoint_tol'], CS['waypoint_tol'], np.deg2rad(CS['angle_tol'])]))
    wps_lo, wps_hi = wps_init - wps_delta, wps_init + wps_delta
    seg_times_lo = from_numpy(np.linalg.norm(wps[1:, :2] - wps[:-1, :2], axis=1) / CS['max_vel'])
    wp_speed_lo = from_numpy(np.array(0))
    trajs_lo, trajs_hi = from_numpy(CS['xyp_lims_lo']), from_numpy(CS['xyp_lims_hi'])

    # define optimizers
    opts = {
        'wps': Adam(wps_init.flatten(), alpha=lr, lo=wps_lo.flatten(), hi=wps_hi.flatten()),
        'seg_times': Adam(seg_times_init, alpha=lr, lo=seg_times_lo),
        'wp_speeds': Adam(wp_speeds_init, alpha=lr, lo=wp_speed_lo)
    }

    n_opts = trange(max_n_opts)
    for n_opt in n_opts:
        wps = opts['wps'].params.view(wps_init.shape)
        seg_times = opts['seg_times'].params
        wp_speeds = opts['wp_speeds'].params

        # compute trajs
        trajs, d_trajs = gen_trajs(wps, wp_speeds, init_vel, n_pts)
        if n_opt == 0:
            trajs_init, d_trajs_init = get_numpy(trajs), get_numpy(d_trajs)

        # compute needed values
        velocities = d_trajs / torch.unsqueeze(torch.unsqueeze(seg_times, 1), 1)

        speeds = torch.norm(velocities, dim=2)
        accelerations = (speeds[:, 1:] - speeds[:, :-1]) / torch.unsqueeze(seg_times, 1) * n_pts

        angles = torch.atan2(d_trajs[:,:,1], d_trajs[:,:,0])
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

        # Compute grad
        grad_wps, grad_seg_times, grad_wp_speeds = grad(loss, [wps, seg_times, wp_speeds])

        grad_wps[torch.isnan(grad_wps)] = 0
        grad_seg_times[torch.isnan(grad_seg_times)] = 0
        grad_wp_speeds[torch.isnan(grad_wp_speeds)] = 0

        # Step optimizer
        opts['wps'].collect_grad(grad_wps.flatten())
        opts['seg_times'].collect_grad(grad_seg_times)
        opts['wp_speeds'].collect_grad(grad_wp_speeds)

        opts['wps'].step()
        opts['seg_times'].step()
        opts['wp_speeds'].step()

        # log progress
        writer.add_scalar('/costs/loss', loss, n_opt)
        writer.add_scalar('/costs/total_time', total_time, n_opt)
        writer.add_scalar('/costs/total_constraints', constraint_cost, n_opt)
        writer.add_scalar('/costs/dynamics', constraint_costs[0], n_opt)
        writer.add_scalar('/costs/steering_angle', constraint_costs[1], n_opt)
        writer.add_scalar('/costs/acceleration', constraint_costs[2], n_opt)
        writer.add_scalar('/costs/speed', constraint_costs[3], n_opt)
        writer.add_scalar('/costs/traj_bounds', constraint_costs[4] + constraint_costs[5], n_opt)
        writer.add_scalar('/grads/wps', grad_wps.mean(), n_opt)
        writer.add_scalar('/grads/seg_times', grad_seg_times.mean(), n_opt)
        writer.add_scalar('/grads/wp_speeds', grad_wp_speeds.mean(), n_opt)

        for i in range(len(seg_times)):
            writer.add_scalar('/seg_times/{}'.format(i), seg_times[i], n_opt)
        for i in range(len(wp_speeds)):
            writer.add_scalar('/wp_speeds/{}'.format(i), wp_speeds[i], n_opt)

        n_opts.set_description('Loss {:.3f} | Time {:.3f} | TC {:.3f} | Dynamics {:.3f}'.format(
            get_numpy(loss), get_numpy(total_time), get_numpy(constraint_cost), get_numpy(constraint_costs[0])
        ))
        n_opts.refresh()

    return {
        'wps': get_numpy(opts['wps'].params.view(wps_init.shape)),
        'wp_speeds': get_numpy(opts['wp_speeds'].params),
        'seg_times': get_numpy(opts['seg_times'].params),
        'trajs_init': trajs_init,
        'd_trajs_init': d_trajs_init,
        'trajs': get_numpy(trajs),
        'd_trajs': get_numpy(d_trajs),
        'velocities': get_numpy(velocities),
        'speeds': get_numpy(speeds),
        'accelerations': get_numpy(accelerations),
        'steering_angles': get_numpy(steering_angles),
        'velocities_pred': get_numpy(velocities_pred),
        'betas': get_numpy(betas),
        'total_time': get_numpy(total_time),
        'loss': get_numpy(loss),
        'constraint_costs': get_numpy(constraint_costs),
        'constraint_cost': get_numpy(constraint_cost)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', '-l', type=str, default='outs')
    parser.add_argument('--tag', '-t', type=str, required=True)
    args = parser.parse_args()

    savedir = os.path.join(args.logdir, args.tag)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

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

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'tb', args.tag))
    res = trajopt(wps, writer,
        n_pts=30, 
        # dynamics, steering angle, acceleration, speed, traj bounds
        constraint_weights=[500, 50, 10, 1, 100, 100], 
        max_n_opts=200, 
        lr=5e-1
    )

    plot_trajs(wps, res['trajs_init'], res['d_trajs_init'], title='{} | Init'.format(args.tag), save=os.path.join(savedir, 'init.png'))
    plot_trajs(wps, res['trajs'], res['d_trajs'], title='{} | Final'.format(args.tag), save=os.path.join(savedir, 'final.png'))

    import IPython; IPython.embed(); exit(0)
