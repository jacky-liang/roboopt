function [obj] = objective(target, link_length, obstacles, x(:,1), x(:,2), x(:,3))

[o, n] = size(link_length);
[obs, o] = size(obstacles);

links = forward_links(link_length, r, p, y);

err = pose_err(links(n, :), target);

min_dists = zeros((n, 1));

for i = 1:n-1
    s0 = links(i, :);
    s1 = links(i + 1, :);

    s01 = s1 - s0;
    s01_norm_squared = s01 * s01;

    dists = zeros((obs, 1));

    for j = 1:obs
        p = obstacles(j, 1:3);
        r = obstacles(j, 4);

        t_hat = (p - s0) * s01 / s01_norm_squared;
        t_star = min(max(t_hat, 0), 1);

        dists(j) = norm(s1 + t_star * s01 - p) - r;
    end

    min_dists(i) = min(dists);
end

obj = err - min(min_dists);

end