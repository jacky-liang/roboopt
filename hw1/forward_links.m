function [links] = forward_links(link_length, r, p, y)

[o, n] = size(link_length);

T_c = [0; 0; 0];
R_c = eye(3);

links = zeros([n+1, 3]);

for j = 1:n
    R_j = eul2rotm([r(j), p(j), y(j)], "xyz");
    T_j = [link_length(j); 0; 0];
    
    R_c = R_c * R_j;
    T_c = R_c * T_j + T_c;
    
    links(j + 1, :) = T_c;
end

end

