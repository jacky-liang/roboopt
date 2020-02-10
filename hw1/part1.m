function [r, p, y] = part1(target, link_length, min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw, obstacles)
%% Function that uses optimization to do inverse kinematics for a snake robot

%%Outputs 
  % [r, p, y] = roll, pitch, yaw vectors of the N joint angles
  %            (N link coordinate frames)
%%Inputs:
    % target: [x, y, z, q0, q1, q2, q3]' position and orientation of the end
    %    effector
    % link_length : Nx1 vectors of the lengths of the links
    % min_xxx, max_xxx are the vectors of the 
    %    limits on the roll, pitch, yaw of each link.
    % limits for a joint could be something like [-pi, pi]
    % obstacles: A Mx4 matrix where each row is [ x y z radius ] of a sphere
    %    obstacle. M obstacles.

% Your code goes here.
[o, n] = size(link_length);

r0 = (rand([n,1]) * 2 - 1).*(max_roll - min_roll)' + min_roll';
p0 = (rand([n,1]) * 2 - 1).*(max_pitch - min_pitch)' + min_pitch';
y0 = (rand([n,1]) * 2 - 1).*(max_yaw - min_yaw)' + min_yaw';
x0 = [r0, p0, y0];

lb = [min_roll min_pitch min_yaw];
ub = [max_roll max_pitch max_yaw];
fun = @(x) objective(target, link_length, obstacles, x(:,1), x(:,2), x(:,3));

lb = [min_roll min_pitch min_yaw];
ub = [max_roll max_pitch max_yaw];

options = optimoptions('fmincon','Algorithm','active-set');
x = fmincon(fun, x0, [], [], [], [], lb, ub, [], options);

% CMA-ES
% x = cmaes(fun, x0, lb, ub);

if fun(x) > 0.01
    disp("Could not reach target exactly.");
end

r = x(:,1);
p = x(:,2);
y = x(:,3);



end