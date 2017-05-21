#include "RSOS.h"

using namespace std;

void one_step(int L, int dh_max, Ran &myran, int *h) {
  while (true) {
    double a = myran.doub();
    int i = int(a * L);
    if (i == L) i = 0;
    int j = i - 1;
    if (j < 0) j = L - 1;
    int dh = abs(h[i] - h[j] + 1);
    if (dh <= dh_max) {
      j = i + 1;
      if (j >= L) j = 0;
      dh = abs(h[i] - h[j] + 1);
      if (dh <= dh_max) {
        h[i] += 1;
        break;
      }
    }
  }
}

void run(int L, int dh_max, int seed, int *ts, int len_ts,
         int *ht, int len_ht) {
  int *h = new int[L];
  Ran myran(seed);
  for (int i = 0; i < L; i++)
    h[i] = 0;
  int pos = 0;
  for (int i = 0; i <= ts[len_ts-1]; i++) {
    if (i == ts[pos]) {
      for (int j = 0; j < L; j++)
        ht[j + pos * L] = h[j];
      pos++;
    }
    one_step(L, dh_max, myran, h);
  }
  delete[] h;
}


