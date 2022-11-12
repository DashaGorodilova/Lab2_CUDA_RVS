#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "wb.h"

__global__ void vecAdd(float* in1, float* in2, float* out, int len)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char** argv)
{
    wbArg_t args;
    int inputLength;
    float* hostInput1;
    float* hostInput2;
    float* hostOutput;
    float* deviceInput1;
    float* deviceInput2;
    float* deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 =
        (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
        (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**)&deviceInput1, inputLength);
    cudaMalloc((void**)&deviceInput2, inputLength);
    cudaMalloc((void**)&deviceOutput, inputLength);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Скопируйте память на GPU
    cudaMemcpy(deviceInput1, hostInput1, inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostOutput, inputLength, cudaMemcpyDeviceToHost);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Инициализация сетки и размеры блока здесь
    dim3 DimGrid((inputLength - 1) / 256 + 1, 1, 1);
    dim3 DimBlock(256, 1, 1);
    vecAdd << <DimGrid, DimBlock >> > (deviceInput1, deviceInput2, deviceOutput, inputLength);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Запустите ядро GPU
    vecAdd << <ceil(inputLength / 256.0), 256 >> > (deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Скопируйте  GPU память обратно на CPU здесь
    cudaMemcpy(deviceInput1, hostInput1, inputLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(deviceInput2, hostInput2, inputLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(deviceOutput, hostOutput, inputLength, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Освободите память GPU
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}