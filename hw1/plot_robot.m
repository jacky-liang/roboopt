function [] = plot_robot(obstacles, target, link_length, r, p, y)
[no, f] = size(obstacles);
links = forward_links(link_length, r, p, y);

figure
hold on

axis equal
[x, y, z] = sphere();

for i=1:no
    r = obstacles(i, 4);
    a = obstacles(i, 1);
    b = obstacles(i, 2);
    c = obstacles(i, 3);
    
    surf(x*r+a, y*r+b, z*r+c)
end

plot3(target(1), target(2), target(3), '*')
plot3(links(:,1), links(:,2), links(:,3), '-o')
view(3)

end
