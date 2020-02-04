target=[1,0,0,1,0,0,0];
link_length=[1, 0.5, 0.3];

min_roll=degtorad([-90,-90,-90]);
max_roll=degtorad([90,90,90]);
min_pitch=degtorad([-90,-90,-90]);
max_pitch=degtorad([90,90,90]);
min_yaw=degtorad([-90,-90,-90]);
max_yaw=degtorad([90,90,90]);

obstacles=[
    1, 1, 1, 0.1;
    -1,-1,-1,0.1
    ];

[r, p, y] = part1(target, link_length, min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw, obstacles)