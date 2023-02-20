# cython: cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#cython: boundscheck=False, wraparound=False
# from libc.math cimport log
# import hazelbean as hb
# import time
# from collections import OrderedDict
# from cython.parallel cimport prange
import cython
cimport cython
import numpy as np  # NOTE, both imports are required. cimport adds extra information to the pyd while the import actually defines numppy
cimport numpy as np
from numpy cimport ndarray
# from libc.math cimport sin
# from libc.math cimport fabs
import math, time
# from cython.view cimport array as cvarray

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def write_carbon_table_to_array(
        ndarray[np.float32_t, ndim=2] lulc_array not None,  # NOTE Funny float usage
        ndarray[np.float32_t, ndim=2] carbon_zones_array not None,  # NOTE Funny float usage
        ndarray[np.float32_t, ndim=2] lookup_table not None,
        dict row_names,
        dict col_names):
    cdef long long cr, cc
    cdef double c_lulc_class, c_carbon_zone  # These are doubles to match float32
    cdef long long n_rows = lulc_array.shape[0]
    cdef long long n_cols = lulc_array.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] output_carbon = np.zeros((n_rows, n_cols), dtype=np.float32)

    start = time.time()
    # print('starting cython loop', start)
    for cr in range(n_rows):
        # if cr % 10 == 0:
            # print(cr)
        for cc in range(n_cols):
            c_lulc_class = lulc_array[cr, cc]
            c_carbon_zone = carbon_zones_array[cr, cc]
            if c_carbon_zone > 0 and c_lulc_class > 0:
                # print('c_lulc_class', c_lulc_class)
                # print('c_carbon_zone', c_carbon_zone)
                # print('c_lulc_class[cr, cc]', c_lulc_class)
                # lookup_r_id =col_names[c_lulc_class]
                # print('c_carbon_zone c_lulc_class', c_carbon_zone, c_lulc_class)
                # print('row_names[c_carbon_zone], col_names[c_lulc_class]', row_names[c_carbon_zone], col_names[c_lulc_class])
                try:
                    output_carbon[cr, cc] = lookup_table[row_names[c_carbon_zone], col_names[c_lulc_class]]
                except:
                    pass
    # print('ending cython loop', str(time.time() - start))
    return output_carbon
