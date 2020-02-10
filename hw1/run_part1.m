load('data.mat');

obstacles=[
    0.5, 0.5, 0.5, 0.25;
    0.5,0.5,-0.5,0.25;
    0.5,-0.5,0.5,0.25;
    0.5,-0.5,-0.5,0.25;
    -0.5,0.5,0.5, 0.25;
    -0.5,0.5,-0.5,0.25;
    -0.5,-0.5,0.5,0.25;
    -0.5,-0.5,-0.5,0.25
    ];

pose_errs = zeros(N, 1);
objs = zeros(N, 1);
times = zeros(N, 1);

for i = 1:N
    target = targets(i, :);
    link_length = link_lengths(i, :);
    
    s = cputime;
    [r_p, p_p, y_p] = part1(target, link_length, min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw, obstacles);
    times(i) = cputime - s;
    
    reached = forward(link_length, r_p, p_p, y_p);
        
%     plot_robot(obstacles, target, link_length, r_p, p_p, y_p);

    objs(i) = objective(target, link_length, obstacles, r_p, p_p, y_p);
    pose_errs(i) = pose_err(target, reached);
end

mean(times)
std(times)

mean(objs)
std(objs)

mean(pose_errs)
std(pose_errs)