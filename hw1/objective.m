function [obj] = objective(target, link_length, obstacles, r, p, y)

[obs, o] = size(obstacles);

links = forward_links(link_length, r, p, y);

[n1, o] = size(links);

err = pose_err(target, links(n1, :));

min_dists = zeros(n1-1, 1);

for i = 1:n1-1
    s0 = links(i, 1:3);
    s1 = links(i + 1, 1:3);

    s01 = s1 - s0;
    s01_norm_squared = s01 * s01';

    dists = zeros(obs, 1);

    for j = 1:obs
        p = obstacles(j, 1:3);
        r = obstacles(j, 4);

        t_hat = (p - s0) * s01' / s01_norm_squared;
        t_star = min(max(t_hat, 0), 1);

        dists(j) = norm(s1 + t_star * s01 - p) - r;
    end

    min_dists(i) = min(dists);
end

obj = err - 0.4 * min(min_dists);

end