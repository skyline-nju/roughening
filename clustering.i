%module clustering
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "clustering.h"
%}
/*  include the numpy typemaps */
%include "../numpy.i"
/*  need this for correct module initialization */
%init %{
    import_array();
%}

/*  typemaps for the two arrays, the second will be modified in-place */
%apply (double* IN_ARRAY1, int DIM1) {(double * in_arr1, int size_in1),
                                      (double * in_arr2, int size_in2)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int * out_arr1, int size_out1),
                                        (int * out_arr2, int size_out2)}

/*  Wrapper for cos_doubles that massages the types */
%rename (DBSCAN) my_DBSCAN;
%inline %{
    /*  takes as input two numpy arrays */
    void my_DBSCAN(double *in_arr1, int size_in1, double *in_arr2, int size_in2,
                   double r0, int min_pts, double Lx, double Ly,
                   int *out_arr1, int size_out1, int *out_arr2, int size_out2) {
      DBSCAN(in_arr1, in_arr2, size_in1, r0, min_pts, Lx, Ly, out_arr1, out_arr2);
    }
%}