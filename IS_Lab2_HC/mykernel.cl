

__kernel void pi_estimator(__global float* Leibniz_data, 
     int elements_per_workitem,
     int iterations,
     __local  float* local_result, 
     __global float* global_result) {


  /* Make sure previous processing has completed */
  barrier(CLK_LOCAL_MEM_FENCE);
       
  int work_item_ID_offset = get_global_id(0) * elements_per_workitem;

// step counting  is the index jump of the input buffer from 1 to the end of iterations
// iteration are the number of the reduction couples x-y  , x and y are the fractions x=1/1 y=-1/3 , x= 1/5 , etc 
for (int step=1; step<=(iterations); step=step*2){

  if((work_item_ID_offset%(step*2))==0) // we check if the workitem we use for each addition is a multiplyer of 2 
  {
    // we replace the original values if the input buffer with new calculate addition 
    Leibniz_data[work_item_ID_offset] = Leibniz_data[work_item_ID_offset] + Leibniz_data[work_item_ID_offset+step];
  }
}

/* Make sure local processing has completed */
barrier(CLK_GLOBAL_MEM_FENCE);



// make sure all work items are done
if (get_local_id(0)==0){
  float PI=0;

  if(work_item_ID_offset==0){
     // Pi calculation comes from the first work-item addition with offset=0 multiplied with 4
      PI = Leibniz_data[work_item_ID_offset]*4;
      printf("\n\n\r Pi Result from Kernel: %f\n",PI);
      global_result[0] = PI; // send the result back to the host via the output global buffer
}}
}