#include <iostream>
#include <math.h>
#include <array>

bool clip_test(double p, double q, double &tenter, double &texit)
{

	if (p == 0)
        {
               	//std::cout << "Parallel line" << endl;
                return q >= 0;
        }
        else
        {
                double t = q / p;
                if (p < 0)
                {
                	if (t > tenter)
                        {
                        	tenter = t;
                        }
                }
                else
                {
                	if (t < texit)
                        {
				texit = t;
                        }
                }
        }

	return tenter <= texit;
}

std::array<double, 6> liang_barsky_3d(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
		double x1, double y1, double z1, double x2, double y2, double z2)
{
        double dx = x2 - x1;
        double dy = y2 - y1;
        double dz = z2 - z1;

	std::array<double, 6> arr = {-999, -999, -999, -999, -999, -999};

	double tenter = 0.0;
        double texit = 1.0;

	if (clip_test(-dx, x1 - xmin, tenter, texit) && clip_test(dx, xmax - x1, tenter, texit) &&
	    clip_test(-dy, y1 - ymin, tenter, texit) && clip_test(dy, ymax - y1, tenter, texit) &&
            clip_test(-dz, z1 - zmin, tenter, texit) && clip_test(dz, zmax - z1, tenter, texit))
	{
		arr[0] = x1 + tenter * dx;
		arr[1] = y1 + tenter * dy;
		arr[2] = z1 + tenter * dz;
		arr[3] = x1 + texit * dx;
		arr[4] = y1 + texit * dy;
		arr[5] = z1 + texit * dz;
	}

        if (tenter > texit)
        {
                //std::cout << "Line completely outside" << endl;
                return arr;
	}        

	//std::cout << arr[0] << ", " << arr[1] << "\n" << arr[2] << ", " << arr[3] << "\n" << arr[4] << ", " << arr[5] << endl;

        return arr;
}

