function [ee] = forward(link_length, r, p, y)

[o, n] = size(link_length);

T_c = [0; 0; 0];
R_c = eye(3);

for j = 1:n
    R_j = eul2rotm([r(j), p(j), y(j)], "xyz");
    T_j = [link_length(j); 0; 0];
    
    R_c = R_c * R_j;
    T_c = R_c * T_j + T_c;
end

ee_tra = T_c';
ee_quat = rotm2quat(R_c);
ee = [ee_tra ee_quat];

end

