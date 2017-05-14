#include "clustering.h"
#include <cstdlib>
#include <cmath>
#include <list>
using namespace std;

struct CellList
{
  vector<int> partical_num;
  static double lx;
  static double ly;
  static double Lx;
  static double Ly;
  static int ncols;
  static int nrows;
};

double CellList::lx;
double CellList::ly;
double CellList::Ly;
double CellList::Lx;
int CellList::ncols;
int CellList::nrows;

int cal_cell_idx(double x, double y) {
  int col = int(x / CellList::lx);
  if (col >= CellList::ncols) col = CellList::ncols - 1;
  int row = int(y / CellList::ly);
  if (row >= CellList::nrows) row = CellList::nrows - 1;
  return col + CellList::ncols * row;
}

void create_cell_list(const double *x, const double *y, int size, double r0,
                      double Lx, double Ly, CellList **cell_list) {
  CellList::nrows = int(ceil(Ly / r0));
  CellList::ncols = int(ceil(Lx / r0));
  CellList::Lx = Lx;
  CellList::Ly = Ly;
  CellList::ly = Ly / CellList::nrows;
  CellList::lx = Lx / CellList::ncols;
  CellList *clist = new CellList[CellList::nrows * CellList::ncols];
  for (int i = 0; i < size; i++) {
    int cell_idx = cal_cell_idx(x[i], y[i]);
    clist[cell_idx].partical_num.push_back(i);
  }
  *cell_list = clist;
}

bool within_r0(double dx, double dy, double r0) {
  if (dx < -0.5 * CellList::Lx)
    dx += CellList::Lx;
  else if (dx > 0.5 * CellList::Lx)
    dx -= CellList::Lx;
  if (dy < -0.5 * CellList::Ly)
    dy += CellList::Ly;
  else if (dy > 0.5 * CellList::Ly)
    dy -= CellList::Ly;
  return (dx * dx + dy * dy) <= r0 * r0;
}

void cal_neighbor_list(int i0, const double *x, const double *y, double r0,
                       const CellList *cell_list, list<int> &neighbor) {
  double x0 = x[i0];
  double y0 = y[i0];
  int cell_idx0 = cal_cell_idx(x0, y0);
  int col0 = cell_idx0 % CellList::ncols;
  int row0 = cell_idx0 / CellList::ncols;
  for (int j = -1; j < 2; j++) {
    int row = row0 + j;
    if (row < 0)
      row += CellList::nrows;
    else if (row >= CellList::nrows)
      row -= CellList::nrows;
    for (int i = -1; i < 2; i++) {
      int col = col0 + i;
      if (col < 0)
        col += CellList::ncols;
      else if (col >= CellList::ncols)
        col -= CellList::ncols;
      int cell_idx = col + row * CellList::ncols;
      for (int k : cell_list[cell_idx].partical_num) {
        if (within_r0(x0 - x[k], y0 - y[k], r0) && k != i0)
          neighbor.push_back(k);
      }
    }
  }
}

void expand_cluster(list<int> &neighbor, bool *visited, bool *clustered,
                    const double *x, const double *y, double r0, int min_pts,
                    const CellList *cell_list, list<int> &cluster) {
  for (auto iter = neighbor.begin(); iter != neighbor.end(); ++iter) {
    int k = *iter;
    if (!visited[k]) {
      visited[k] = true;
      list<int> neighbor_new;
      cal_neighbor_list(k, x, y, r0, cell_list, neighbor_new);
      if (neighbor_new.size() >= min_pts)
        neighbor.splice(neighbor.end(), neighbor_new);
    }
    if (!clustered[k]) {
      clustered[k] = true;
      cluster.push_back(k);
    }
  }
}

void DBSCAN(const double *x, const double *y, int size,
            double r0, int min_pts, double Lx, double Ly,
            int *num_arr, int *sep) {
  CellList *cell_list = nullptr;
  create_cell_list(x, y, size, r0, Lx, Ly, &cell_list);
  bool *visited = new bool[size];
  bool *clustered = new bool[size];
  for (int i = 0; i < size; i++) {
    visited[i] = false;
    clustered[i] = false;
  }
  int pos_num_arr = 0;
  int pos_sep = 0;
  for (int i = 0; i < size; i++) {
    if (!visited[i]) {
      visited[i] = true;
      list<int> neighbor;
      cal_neighbor_list(i, x, y, r0, cell_list, neighbor);
      if (neighbor.size() >= min_pts) {
        list<int> cluster = { i };
        clustered[i] = true;
        expand_cluster(neighbor, visited, clustered, x, y, r0, min_pts,
                       cell_list, cluster);
        for (auto particle_num : cluster) {
          num_arr[pos_num_arr] = particle_num;
          pos_num_arr++;
        }
        sep[pos_sep] = pos_num_arr;
        pos_sep += 1;
      }
    }
  }
  for (int i = 0; i < size; i++) {
    if (!clustered[i]) {
      num_arr[pos_num_arr] = i;
      pos_num_arr++;
      sep[pos_sep] = pos_num_arr;
      pos_sep += 1;
    }
  }
  delete[] cell_list;
  delete[] visited;
  delete[] clustered;
}