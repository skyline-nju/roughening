%module RSOS
%{
  #define SWIG_FILE_WITH_INIT
  #include "RSOS.h"
%}

%include "../numpy.i"


%init %{
  import_array();
%}

%apply (int *IN_ARRAY1, int DIM1) {(int *ts, int len_ts)}
%apply (int *INPLACE_ARRAY1, int DIM1) {(int *ht, int len_ht)}

%include "RSOS.h"