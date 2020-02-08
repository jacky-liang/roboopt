function [err] = pose_err(target, actual)

delta_tra = norm(target(1:3) - actual(1:3));

actual_quat = quaternion(actual(4), actual(5), actual(6), actual(7));
target_quat = quaternion(target(4), target(5), target(6), target(7));

actual_R = quat2rotm(actual_quat);
target_R = quat2rotm(target_quat);
delta_R = actual_R * target_R';

delta_aa = rotm2axang(delta_R);
delta_angle = delta_aa(4);

err = delta_tra^2 + delta_angle^2;

end