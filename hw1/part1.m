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

x0 = zeros(n, 3);
lb = [min_roll min_pitch min_yaw];
ub = [max_roll max_pitch max_yaw];
fun = @(x) objective(target, link_length, obstacles, x(:,1), x(:,2), x(:,3));

A = zeros(n * 3);
b = ones(n * 3, 1) * 1000;
Aeq = zeros(n * 3);
beq = zeros(n * 3, 1);
lb = [min_roll min_pitch min_yaw];
ub = [max_roll max_pitch max_yaw];

x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub);

if fun(x) > 0.01
    disp("Could not reach target exactly.");
end

r = x(:,1);
p = x(:,2);
y = x(:,3);

end