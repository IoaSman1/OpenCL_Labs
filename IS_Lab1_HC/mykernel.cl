/* widthA=heightB for valid matrix multiplication */
__kernel void simpleMultiply(
    __global float *outputD,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    int widthC,
    int heightC,
    __global float *inputA,
    __global float *inputB,
    __global float *inputC)
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    
    /* get global position in X direction */
    int col = get_global_id (0);

    float sum = 0.0f;

    /* calculate result of one element of Matrix C */
    for (int i=0; i<widthA; i++) {
        //printf("index: %d row: %d col: %d\n",i, row, col);
        sum += inputA[row*widthA + i] * inputB[i*widthB + col];
        //printf("sum: %f\n",sum);
         //int r =row*widthA + i;
         //int c = i*widthB + col;
         //printf("r: %d\n",r);
         //printf("c: %d\n",c);
    }

      // printf("sum: %f\n",sum);
    outputD[row*widthB + col] = sum + inputC[col];

}
