#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <vector>
void DBSCAN(const double *x, const double *y, int size,
            double r0, int min_pts, double Lx, double Ly,
            int *num_arr, int *sep);

struct CellList
{
  std::vector<int> partical_num;
  static double lx;
  static double ly;
  static double Lx;
  static double Ly;
  static int ncols;
  static int nrows;
};

#endif