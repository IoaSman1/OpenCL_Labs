/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "mykernel.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[1] = {
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 1, initBlocks, pair_map };

// Block function
void (^pi_estimator_kernel)(const cl_ndrange *ndrange, cl_float* Leibniz_data, cl_int elements_per_workitem, cl_int iterations, size_t local_result, cl_float* global_result) =
^(const cl_ndrange *ndrange, cl_float* Leibniz_data, cl_int elements_per_workitem, cl_int iterations, size_t local_result, cl_float* global_result) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel pi_estimator does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, Leibniz_data, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(elements_per_workitem), &elements_per_workitem, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(iterations), &iterations, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, local_result, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, global_result, &kargs);
  gcl_log_cl_fatal(err, "setting argument for pi_estimator failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing pi_estimator failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("mykernel.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == pi_estimator_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "pi_estimator", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = pi_estimator_kernel;
}

