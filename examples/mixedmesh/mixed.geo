// Gmsh project created on Wed Oct 19 17:08:55 2022
SetFactory("OpenCASCADE");
//+
Point(1) = {0.0, 0.0, 0.0, 1.0};
//+
Point(2) = {1.0, 0.0, 0.0, 1.0};
//+
Point(3) = {1.0, 1.0, 0.0, 1.0};
//+
Point(4) = {0.0, 1.0, 0.0, 1.0};
//+
Point(5) = {0.0, 0.0, 1.0, 1.0};
//+
Point(6) = {1.0, 0.0, 1.0, 1.0};
//+
Point(7) = {1.0, 1.0, 1.0, 1.0};
//+
Point(8) = {0.0, 1.0, 1.0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 5};
//+
Line(9) = {1, 5};
//+
Line(10) = {2, 6};
//+
Line(11) = {3, 7};
//+
Line(12) = {4, 8};
//+
Curve Loop(1) = {1,2,3,4};
//+
Curve Loop(2) = {1,10,-5, -9};
//+
Curve Loop(3) = {2,11,-6,-10};
//+
Curve Loop(4) = {3,12,-7,-11};
//+
Curve Loop(5) = {4,9,-8,-12};
//+
Curve Loop(6) = {5,6,7,8};
//+
Plane Surface(1) = {1};
//+
Plane Surface(2) = {2};
//+
Plane Surface(3) = {3};
//+
Plane Surface(4) = {4};
//+
Plane Surface(5) = {5};
//+
Plane Surface(6) = {6};
//+
Surface Loop(1) = {5, 1, 2, 3, 4, 6};
//+
Point(9) = {0.5, 0.5, 1.5, 1.0};
//+
Line(13) = {7, 9};
//+
Line(14) = {6, 9};
//+
Line(15) = {5, 9};
//+
Line(16) = {8, 9};
//+
Curve Loop(7) = {13, -16, -7};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {13, -14, 6};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {14, -15, 5};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {15, -16, 8};
//+
Plane Surface(10) = {10};
//+
Surface Loop(2) = {7, 8, 9, 10, 6};
//+
Volume(1) = {2};
//+
Surface Loop(3) = {4, 1, 2, 3, 5, 6};
//+
Volume(2) = {3};
