#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "rand.h"

using namespace std;

class RSOS
{
public:
  RSOS(int L0, int N0, int seed);
  ~RSOS();
  void deposit();
  void eval(int t, int dt);

private:
  int L;
  int N;
  int *h;
  Ran *myran;
  ofstream fout;
};

RSOS::RSOS(int L0, int N0, int seed): L(L0), N(N0){
  myran = new Ran(seed);
  h = new int[L];
  for (int i = 0; i < L; i++)
    h[i] = 0;
  char buff[100];
  snprintf(buff, 100, "RSOS_%d_%d_%d.bin", L, N, seed);
  fout.open(buff, ios::binary);
}

RSOS::~RSOS(){
  fout.close();
  delete[] h;
}

void RSOS::deposit(){
  while(true){
    double a = myran->doub();
    int i = int(a * L);
    if (i == L) i = 0;
    int j = i - 1;
    if (j < 0) j = L - 1;
    int dh = abs(h[i] - h[j] + 1);
    if (dh <= N){
      j = i + 1;
      if (j >= L) j = 0;
      dh = abs(h[i] - h[j] + 1);
      if (dh <= N){
        h[i] += 1;
        break;
      }
    }
  }
}

void RSOS::eval(int t, int dt){
  int i = 0;
  while (i < t){
    if (i % dt == 0){
      fout.write((char *)h, L * sizeof(int));
      cout << "t = " << i << endl;
    }
    deposit();
    i++;
  }
}

int main(int argc, char* argv[])
{
  int L = atoi(argv[1]);
  int N = atoi(argv[2]);
  int seed = atoi(argv[3]);
  int t = atoi(argv[4]);
  int dt = argc == 6 ? atoi(argv[5]) : 50000;
  RSOS rsos(L, N, seed);
  rsos.eval(t, dt);
}
