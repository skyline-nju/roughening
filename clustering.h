#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <vector>
void DBSCAN(const double *x, const double *y, int size,
            double r0, int min_pts, double Lx, double Ly,
            int *num_arr, int *sep);
#endif