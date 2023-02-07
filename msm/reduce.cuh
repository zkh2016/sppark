#ifndef REDUCE_CUH
#define REDUCE_CUH

#define FINAL_MASK 0xffffffff
#define HALF_WARP 16
#define WARP_SIZE 32

#ifdef __CUDA_ARCH__
template <typename T>
__inline__ __device__ T WarpReduceSum(T val, unsigned lane_mask) {
    for (int mask = HALF_WARP; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(lane_mask, val, mask, warpSize);
    return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T BlockReduceSum(T val, unsigned mask) {
    __shared__ T shared[WARP_SIZE];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduceSum<T>(val, mask);

    __syncthreads();
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    // align block_span to warpSize
    int block_span = (blockDim.x + warpSize - 1) >> 5;
    val = (lane < block_span) ? shared[lane] :
        static_cast<T>(0.0f);
    val = WarpReduceSum<T>(val, mask);

    return val;
}

#endif
#endif
