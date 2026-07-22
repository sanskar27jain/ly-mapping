#ifndef LIANG_BARSKY_H
#define LIANG_BARSKY_H

bool clip_test(double p, double q, double &tenter, double &texit);
std::array<double, 6> liang_barsky_3d(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, 
		double x1, double y1, double z1, double x2, double y2, double z2);

#endif
