#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
  #else
  #include <CL/cl.h>
#endif

#ifdef AOCL
  #include "CL/opencl.h"
  #include "AOCLUtils/aocl_utils.h"
  using namespace aocl_utils;
  void cleanup();
#endif



// ------ Prototypes -------------
void elapsed_T(clock_t start, clock_t end);




#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];

static float A[8] = {
  1.0f,  1.0f,  1.0f,  1.0f,
  1.0f,  1.0f,  1.0f,  1.0f};

static float B[24] = {
  2.0f,  2.0f,  2.0f,  2.0f, 2.0f, 2.0f,
  2.0f,  2.0f,  2.0f,  2.0f, 2.0f, 2.0f,
  2.0f,  2.0f,  2.0f,  2.0f, 2.0f, 2.0f,
  2.0f,  2.0f,  2.0f,  2.0f, 2.0f, 2.0f};

static float C[12] = {
  3.0f,  3.0f,  3.0f,  3.0f, 3.0f, 3.0f,
  3.0f,  3.0f,  3.0f,  3.0f, 3.0f, 3.0f};








int main()
{
    cl_uint           platformCount;
    cl_platform_id*   platforms;
    cl_device_id      device_id;
    cl_uint           ret_num_devices;
    cl_int            ret;
    cl_context        context = NULL;
    cl_command_queue  command_queue = NULL;
    cl_program        program = NULL;
    cl_kernel         kernel = NULL;

    FILE    *fp;
    char    fileName[] = "./mykernel.cl";
    char    *source_str;
    size_t  source_size;

    int wA=4;
    int hA=2;
    int wB=6;
    int hB=4;
    int wC = 6;
    int hC = 2;

    int wD = wC;
    int hD = hC;


clock_t Start_T = clock();


#ifdef __APPLE__
    system("@cls||clear");
     printf("Lab 1 :  \tHeterogenous Computing\n");
     printf("Student:  \tIoannis Smanis \n");
     printf("Date:  \t\t10/29/2018 \n");
    /* Get Platform and Device Info */
    clGetPlatformIDs(1, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
    printf("Platform Count: %d\n",platformCount);
    // we only use platform 0, even if there are more plantforms
    // Query the available OpenCL device.
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, DEVICE_NAME_LEN, dev_name, NULL);
    printf("Device Name: \t %s\n\n", dev_name);
#else



#ifdef AOCL  /* Altera FPGA */
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    // Get the OpenCL platform.
    platforms[0] = findPlatform("Intel(R) FPGA");
    if(platforms[0] == NULL) {
      printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
      return false;
    }
    // Query the available OpenCL device.
    getDevices(platforms[0], CL_DEVICE_TYPE_ALL, &ret_num_devices);
    printf("Platform: %s\n", getPlatformName(platforms[0]).c_str());
    printf("Using one out of %d device(s)\n", ret_num_devices);
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("Device Name =  %s\n", getDeviceName(device_id).c_str());
#else
#error "unknown OpenCL SDK environment"
#endif
#endif



    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue for specific target device */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);





#ifdef __APPLE__
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
    }

    source_str  = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    printf("CL Source size:\t%d  bytes\n",source_size);
    fclose(fp);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,(const size_t *)&source_size, &ret);
    printf("Kernel Created: %d\n\n", program);

    if (ret != CL_SUCCESS) {
      printf("Failed to create program from source.\n");
      exit(1);
    }
#else



#ifdef AOCL  /* on FPGA we need to create kernel from binary */
   /* Create Kernel Program from the binary */
   std::string binary_file = getBoardBinaryFile("mykernel", device_id);
   printf("Using AOCX: %s\n", binary_file.c_str());
   program = createProgramFromBinary(context, binary_file.c_str(), &device_id, 1);
#else
#error "unknown OpenCL SDK environment"
#endif

#endif







    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
      printf("Failed to build program.\n");
      exit(1);
    }

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "simpleMultiply", &ret);
    if (ret != CL_SUCCESS) {
      printf("Failed to create kernel.\n");
      exit(1);
    }

    float *D = (float *)calloc (hD * wD ,  sizeof(float));
    printf ("\n------------ Output Matrix D (Initial State) ------------\n");
    for (int i = 0; i < wD*hD; i++) {
        if (i == wD*hD/2 )printf("\n");
        printf ("%f ", D[i]);
    }
    printf("\n--------------------------------------------------------- \n\n");



    /* We assume A, B, C are float arrays which have been declared and initialized */
    /* allocate space for Matrix A on the device */
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, wA*hA*sizeof(float), NULL, &ret);    
    /* copy Matrix A to the device */
    clEnqueueWriteBuffer(command_queue, bufferA, CL_TRUE, 0, wA*hA*sizeof(float), (void *)A, 0, NULL, NULL);


    /* allocate space for Matrix B on the device */
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, wB*hB*sizeof(float), NULL, &ret);
    /* copy Matrix B to the device */
    clEnqueueWriteBuffer(command_queue, bufferB, CL_TRUE, 0, wB*hB*sizeof(float), (void *)B, 0, NULL, NULL);


    /* allocate space for Matrix C on the device */
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_READ_ONLY, wC*hC*sizeof(float), NULL, &ret);
    /* copy Matrix C to the device */
    clEnqueueWriteBuffer(command_queue, bufferC, CL_TRUE, 0, wC*hC*sizeof(float), (void *)C, 0, NULL, NULL);


    /* allocate space for Matrix D on the device */
    cl_mem bufferD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, wD*hD*sizeof(float), NULL, &ret);



    /* Set the kernel arguments */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferD);
    clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&wA);
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&hA);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&wB);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&hB);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&wC);
    clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&hC);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&bufferA);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&bufferB);
    clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&bufferC);



    /* Execute the kernel */
    size_t globalws[2]={wD, hD};
    size_t localws[2] = {2, 2};
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalws, localws, 0, NULL, NULL);
    
    /* it is important to check the return value.
    for example, when enqueueNDRangeKernel may fail when Work group size
    does not divide evenly into global work size */
    if (ret != CL_SUCCESS) {
      printf("Failed to enqueueNDRangeKernel.\n");
      exit(1);
    }

    /* Copy the output data back to the host */
    clEnqueueReadBuffer(command_queue, bufferD, CL_TRUE, 0, wD*hD*sizeof(float), (void *)D, 0, NULL, NULL);

    /* Verify result */
    printf ("\n------------ Output Matrix D (Resutls) ------------------\n");
    for (int i = 0; i < wD*hD; i++) {
        if (i == wD*hD/2 )printf("\n");
        printf ("%f ", D[i]);
    }
    printf("\n---------------------------------------------------------- \n");


    /* free resources */
    free(D);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseMemObject(bufferD);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    clock_t End_T = clock();
    elapsed_T(Start_T, End_T);
    return 0;
}






#ifdef AOCL
// Altera OpenCL needs this callback function implemented in main.c
// Free the resources allocated during initialization
void cleanup() {
}
#endif


void elapsed_T(clock_t start, clock_t end) {
    float elapsed_time = (float)(end - start) / (float)CLOCKS_PER_SEC;
    printf("\n\nElapsed time: %f seconds\n\n", elapsed_time);
}


