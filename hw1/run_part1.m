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

n = 7;
for i = 1:10
    link_length = rand([1,n]);   
    
    min_roll = deg2rad(ones([1, n]) * -180);
    min_pitch = deg2rad(ones([1, n]) * -180);
    min_yaw = deg2rad(ones([1, n]) * -180);
    max_roll = deg2rad(ones([1, n]) * 180);
    max_pitch = deg2rad(ones([1, n]) * 180);
    max_yaw = deg2rad(ones([1, n]) * 180);
    
    r = (rand([n,1]) * 2 - 1).*(max_roll - min_roll)' + min_roll';
    p = (rand([n,1]) * 2 - 1).*(max_pitch - min_pitch)' + min_pitch';
    y = (rand([n,1]) * 2 - 1).*(max_yaw - min_yaw)' + min_yaw';
    target = forward(link_length, r, p, y);
    
%     plot_robot(obstacles, target, link_length, r, p, y);
        
    [r_p, p_p, y_p] = part1(target, link_length, min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw, obstacles);
    reached = forward(link_length, r_p, p_p, y_p);
    
    plot_robot(obstacles, target, link_length, r_p, p_p, y_p);

    fprintf("Pose Error: %f\n", pose_err(target, reached));
    fprintf("Joint Error: %f\n", norm([r p y] - [r_p p_p y_p], 'fro'));
end

