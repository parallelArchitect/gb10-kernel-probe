// gemm_probe.cu v2 -- gb10-kernel-probe
// Single binary CUTLASS GEMM probe. Runtime config dispatch.
// Build (Pascal): nvcc -O2 -std=c++17 -arch=sm_61 -I$HOME/opt_cuda/cutlass/include -I$HOME/opt_cuda/cutlass/tools/util/include gemm_probe.cu -o gemm_probe -lcudart

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using GemmA = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm61, cutlass::gemm::GemmShape<64,64,8>, cutlass::gemm::GemmShape<32,32,8>, cutlass::gemm::GemmShape<1,1,1>, cutlass::epilogue::thread::LinearCombination<float,1,float,float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using GemmB = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm61, cutlass::gemm::GemmShape<128,128,8>, cutlass::gemm::GemmShape<64,32,8>, cutlass::gemm::GemmShape<1,1,1>, cutlass::epilogue::thread::LinearCombination<float,1,float,float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using GemmC = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm61, cutlass::gemm::GemmShape<128,256,8>, cutlass::gemm::GemmShape<64,64,8>, cutlass::gemm::GemmShape<1,1,1>, cutlass::epilogue::thread::LinearCombination<float,1,float,float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using GemmD = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm61, cutlass::gemm::GemmShape<256,128,8>, cutlass::gemm::GemmShape<64,64,8>, cutlass::gemm::GemmShape<1,1,1>, cutlass::epilogue::thread::LinearCombination<float,1,float,float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using GemmE = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm61, cutlass::gemm::GemmShape<64,128,8>, cutlass::gemm::GemmShape<32,64,8>, cutlass::gemm::GemmShape<1,1,1>, cutlass::epilogue::thread::LinearCombination<float,1,float,float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using GemmF = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm61, cutlass::gemm::GemmShape<128,64,8>, cutlass::gemm::GemmShape<64,32,8>, cutlass::gemm::GemmShape<1,1,1>, cutlass::epilogue::thread::LinearCombination<float,1,float,float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

template <typename GemmKernel>
double run_gemm(int M, int N, int K, int warmup, int iters,
                cutlass::Status &status_out, int *smem_out, float *occ_out) {
    cutlass::HostTensor<float, cutlass::layout::RowMajor> A({M,K});
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B({K,N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C({M,N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> D({M,N});
    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 42, 1.0f, -1.0f);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 43, 1.0f, -1.0f);
    cutlass::reference::host::TensorFill(C.host_view(), 0.0f);
    A.sync_device(); B.sync_device(); C.sync_device();
    typename GemmKernel::Arguments args({M,N,K}, A.device_ref(), B.device_ref(), C.device_ref(), D.device_ref(), {1.0f, 0.0f});
    GemmKernel gemm_op;
    status_out = gemm_op.initialize(args);
    if (status_out != cutlass::Status::kSuccess) return -1.0;
    for (int i = 0; i < warmup; i++) { status_out = gemm_op.run(); if (status_out != cutlass::Status::kSuccess) return -1.0; }
    cudaDeviceSynchronize();
    double t0 = get_time_ms();
    for (int i = 0; i < iters; i++) gemm_op.run();
    cudaDeviceSynchronize();
    double elapsed_ms = get_time_ms() - t0;
    *smem_out = 0; *occ_out = 0.0f;
    cudaFuncAttributes attrs;
    auto kfn = cutlass::Kernel<typename GemmKernel::GemmKernel>;
    if (cudaFuncGetAttributes(&attrs, (const void*)kfn) == cudaSuccess) {
        *smem_out = (int)attrs.sharedSizeBytes;
        int mb = 0, dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, (const void*)kfn, GemmKernel::GemmKernel::kThreadCount, attrs.sharedSizeBytes);
        int mw = prop.maxThreadsPerMultiProcessor / 32;
        int kw = GemmKernel::GemmKernel::kThreadCount / 32;
        *occ_out = mb > 0 ? (float)(mb * kw) / mw : 0.0f;
    }
    return (2.0 * M * N * K * iters) / (elapsed_ms * 1e9);
}

int main(int argc, char **argv) {
    int tb_m=128, tb_n=128, tb_k=32, stages=2, warmup=10, iters=100, M=4096, N=4096, K=4096;
    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i],"--tb-m")   && i+1<argc) tb_m   = atoi(argv[++i]);
        if (!strcmp(argv[i],"--tb-n")   && i+1<argc) tb_n   = atoi(argv[++i]);
        if (!strcmp(argv[i],"--tb-k")   && i+1<argc) tb_k   = atoi(argv[++i]);
        if (!strcmp(argv[i],"--stages") && i+1<argc) stages  = atoi(argv[++i]);
        if (!strcmp(argv[i],"--warmup") && i+1<argc) warmup  = atoi(argv[++i]);
        if (!strcmp(argv[i],"--iters")  && i+1<argc) iters   = atoi(argv[++i]);
        if (!strcmp(argv[i],"--m")      && i+1<argc) M       = atoi(argv[++i]);
        if (!strcmp(argv[i],"--n")      && i+1<argc) N       = atoi(argv[++i]);
        if (!strcmp(argv[i],"--k")      && i+1<argc) K       = atoi(argv[++i]);
    }
    cutlass::Status status = cutlass::Status::kSuccess;
    double tflops = -1.0;
    const char *cfg = "unknown";
    int smem = 0; float occ = 0.0f;
    if      (tb_m==64  && tb_n==64)  { cfg="F32_64x64";   tflops=run_gemm<GemmA>(M,N,K,warmup,iters,status,&smem,&occ); }
    else if (tb_m==64  && tb_n==128) { cfg="F32_64x128";  tflops=run_gemm<GemmE>(M,N,K,warmup,iters,status,&smem,&occ); }
    else if (tb_m==128 && tb_n==64)  { cfg="F32_128x64";  tflops=run_gemm<GemmF>(M,N,K,warmup,iters,status,&smem,&occ); }
    else if (tb_m==128 && tb_n==128) { cfg="F32_128x128"; tflops=run_gemm<GemmB>(M,N,K,warmup,iters,status,&smem,&occ); }
    else if (tb_m==128 && tb_n==256) { cfg="F32_128x256"; tflops=run_gemm<GemmC>(M,N,K,warmup,iters,status,&smem,&occ); }
    else if (tb_m==256 && tb_n==128) { cfg="F32_256x128"; tflops=run_gemm<GemmD>(M,N,K,warmup,iters,status,&smem,&occ); }
    else { fprintf(stderr,"No config for tb=%dx%d\n",tb_m,tb_n); }
    const char *st = (status==cutlass::Status::kSuccess && tflops>0) ? "pass" : "fail";
    printf("{\"tb_shape\":\"%dx%dx%d\",\"stages\":%d,\"M\":%d,\"N\":%d,\"K\":%d,\"tflops\":%.4f,\"smem_bytes\":%d,\"occupancy\":%.4f,\"config_name\":\"%s\",\"run_status\":\"%s\"}\n",
           tb_m,tb_n,tb_k,stages,M,N,K,tflops,smem,occ,cfg,st);
    return (status==cutlass::Status::kSuccess && tflops>0) ? 0 : 1;
}
