n = 7;
N = 10;

link_lengths = zeros(N, n);
targets = zeros(N, 7);

min_roll = deg2rad(ones([1, n]) * -180);
min_pitch = deg2rad(ones([1, n]) * -180);
min_yaw = deg2rad(ones([1, n]) * -180);
max_roll = deg2rad(ones([1, n]) * 180);
max_pitch = deg2rad(ones([1, n]) * 180);
max_yaw = deg2rad(ones([1, n]) * 180);

for i = 1:N
    link_lengths(i, :) = rand([1,n]);
    
    r = (rand([n,1]) * 2 - 1).*(max_roll - min_roll)' + min_roll';
    p = (rand([n,1]) * 2 - 1).*(max_pitch - min_pitch)' + min_pitch';
    y = (rand([n,1]) * 2 - 1).*(max_yaw - min_yaw)' + min_yaw';
    
    targets(i, :) = forward(link_lengths(i, :), r, p, y);
    
%     plot_robot(obstacles, target, link_length, r, p, y);
end

save('data.mat'); 