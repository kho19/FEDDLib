SetFactory("OpenCASCADE");

lc = 0.015;
Point(1) = {0,0,0, lc};
Point(2) = {5,0,0, lc};
Point(3) = {5,1,0, lc};
Point(4) = {0,1,0, lc};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};
Physical Surface(10) = {1};
Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
