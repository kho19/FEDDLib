Mesh.ElementOrder = 1;
Mesh.Format = 1;
Mesh.MshFileVersion = 2.2;
h = .02;

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(10) = {1};
Physical Curve(1) = {1, 2, 3, 4};
Physical Surface(10) = {10};
