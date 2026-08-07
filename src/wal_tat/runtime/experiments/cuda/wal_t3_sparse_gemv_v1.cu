#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <int Threads>
__inline__ __device__ float block_sum(float value) {
  __shared__ float sums[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) sums[warp] = value;
  __syncthreads();
  value = threadIdx.x < Threads / 32 ? sums[lane] : 0.0f;
  if (warp == 0) value = warp_sum(value);
  return value;
}

template <int Threads>
__global__ void t3_sparse_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ t3,
    const uint8_t* __restrict__ positions,
    const uint8_t* __restrict__ sign_bits,
    const __half* __restrict__ alpha,
    const __half* __restrict__ beta,
    float* __restrict__ output,
    int rows,
    int columns,
    int groups) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int bytes_per_row = columns >> 2;
  float value = 0.0f;

  for (int byte = threadIdx.x; byte < bytes_per_row; byte += Threads) {
    const int group = byte >> 5;
    const int k = byte << 2;
    const uint8_t packed = t3[row * bytes_per_row + byte];
    const float scale = __half2float(alpha[row * groups + group]);
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 4; lane += 2) {
      const int code0 = static_cast<int>((packed >> (lane * 2)) & 3) - 1;
      const int code1 = static_cast<int>((packed >> ((lane + 1) * 2)) & 3) - 1;
      const __nv_bfloat162 packed_input =
          *reinterpret_cast<const __nv_bfloat162*>(x + k + lane);
      const float2 input = __bfloat1622float2(packed_input);
      value += scale * static_cast<float>(code0) * input.x;
      value += scale * static_cast<float>(code1) * input.y;
    }
#else
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const int code = static_cast<int>((packed >> (lane * 2)) & 3) - 1;
      value += scale * static_cast<float>(code) * __bfloat162float(x[k + lane]);
    }
#endif
  }

  const int sparse_per_row = groups * 8;
  for (int index = threadIdx.x; index < sparse_per_row; index += Threads) {
    const int group = index >> 3;
    const int slot = index & 7;
    const int position = positions[(row * groups + group) * 8 + slot];
    const uint8_t bits = sign_bits[row * groups + group];
    const float sign = ((bits >> slot) & 1) ? 1.0f : -1.0f;
    value += __half2float(beta[row * groups + group]) * sign *
             __bfloat162float(x[group * 128 + position]);
  }

  value = block_sum<Threads>(value);
  if (threadIdx.x == 0) output[row] = value;
}

}  // namespace

torch::Tensor t3_sparse_gemv_cuda(
    torch::Tensor x,
    torch::Tensor t3,
    torch::Tensor positions,
    torch::Tensor sign_bits,
    torch::Tensor alpha,
    torch::Tensor beta,
    int64_t rows,
    int64_t columns,
    int64_t threads) {
  TORCH_CHECK(x.is_cuda() && t3.is_cuda() && positions.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(sign_bits.is_cuda() && alpha.is_cuda() && beta.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(t3.scalar_type() == torch::kUInt8, "t3 must be uint8");
  TORCH_CHECK(positions.scalar_type() == torch::kUInt8, "positions must be uint8");
  TORCH_CHECK(sign_bits.scalar_type() == torch::kUInt8, "sign bits must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat16 && beta.scalar_type() == torch::kFloat16,
              "scales must be float16");
  TORCH_CHECK(x.is_contiguous() && t3.is_contiguous() && positions.is_contiguous(),
              "contiguous tensors required");
  TORCH_CHECK(columns % 128 == 0 && x.numel() == columns, "invalid shape");
  const int groups = static_cast<int>(columns / 128);
  auto output = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 128) {
    t3_sparse_gemv_kernel<128><<<rows, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        t3.data_ptr<uint8_t>(), positions.data_ptr<uint8_t>(),
        sign_bits.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(alpha.data_ptr()),
        reinterpret_cast<const __half*>(beta.data_ptr()), output.data_ptr<float>(),
        static_cast<int>(rows), static_cast<int>(columns), groups);
  } else if (threads == 256) {
    t3_sparse_gemv_kernel<256><<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        t3.data_ptr<uint8_t>(), positions.data_ptr<uint8_t>(),
        sign_bits.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(alpha.data_ptr()),
        reinterpret_cast<const __half*>(beta.data_ptr()), output.data_ptr<float>(),
        static_cast<int>(rows), static_cast<int>(columns), groups);
  } else {
    TORCH_CHECK(false, "threads must be 128 or 256");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("t3_sparse_gemv", &t3_sparse_gemv_cuda, "WAL T3+sparse-k8 GEMV");
}
