#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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
__inline__ __device__ void block_sum_triple(float& a, float& b, float& c) {
  __shared__ float sa[Threads / 32];
  __shared__ float sb[Threads / 32];
  __shared__ float sc[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  a = warp_sum(a); b = warp_sum(b); c = warp_sum(c);
  if (lane == 0) { sa[warp] = a; sb[warp] = b; sc[warp] = c; }
  __syncthreads();
  a = threadIdx.x < Threads / 32 ? sa[lane] : 0.0f;
  b = threadIdx.x < Threads / 32 ? sb[lane] : 0.0f;
  c = threadIdx.x < Threads / 32 ? sc[lane] : 0.0f;
  if (warp == 0) { a = warp_sum(a); b = warp_sum(b); c = warp_sum(c); }
}

template <int Threads>
__global__ void t3_sparse_batched_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ t3,
    const uint8_t* __restrict__ positions,
    const uint8_t* __restrict__ sign_bits,
    const __half* __restrict__ alpha,
    const __half* __restrict__ beta,
    float* __restrict__ output,
    int rows, int columns, int groups) {
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  if (row >= rows) return;
  x += batch * columns;
  const int bytes_per_row = columns >> 2;
  float value = 0.0f;
  for (int byte = threadIdx.x; byte < bytes_per_row; byte += Threads) {
    const int group = byte >> 5;
    const int k = byte << 2;
    const uint8_t packed = t3[row * bytes_per_row + byte];
    const float scale = __half2float(alpha[row * groups + group]);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const int code = static_cast<int>((packed >> (lane * 2)) & 3) - 1;
      value += scale * static_cast<float>(code) * __bfloat162float(x[k + lane]);
    }
  }
  for (int index = threadIdx.x; index < groups * 8; index += Threads) {
    const int group = index >> 3;
    const int slot = index & 7;
    const int position = positions[(row * groups + group) * 8 + slot];
    const uint8_t bits = sign_bits[row * groups + group];
    const float sign = ((bits >> slot) & 1) ? 1.0f : -1.0f;
    value += __half2float(beta[row * groups + group]) * sign *
             __bfloat162float(x[group * 128 + position]);
  }
  value = block_sum<Threads>(value);
  if (threadIdx.x == 0) output[batch * rows + row] = value;
}

template <int Threads>
__global__ void binary_pair_v_batched_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ bits0, const uint8_t* __restrict__ bits1,
    const __half* __restrict__ latent0, const __half* __restrict__ latent1,
    const __half* __restrict__ column0, const __half* __restrict__ column1,
    float* __restrict__ hidden0, float* __restrict__ hidden1,
    int rank, int columns) {
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  if (row >= rank) return;
  x += batch * columns;
  const int bytes_per_row = columns >> 3;
  float sum0 = 0.0f, sum1 = 0.0f;
  for (int byte = threadIdx.x; byte < bytes_per_row; byte += Threads) {
    const uint8_t packed0 = bits0[row * bytes_per_row + byte];
    const uint8_t packed1 = bits1[row * bytes_per_row + byte];
    const int k = byte << 3;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const float input = __bfloat162float(x[k + lane]);
      sum0 += (((packed0 >> lane) & 1) ? 1.0f : -1.0f) * input *
              __half2float(column0[k + lane]);
      sum1 += (((packed1 >> lane) & 1) ? 1.0f : -1.0f) * input *
              __half2float(column1[k + lane]);
    }
  }
  sum0 = block_sum<Threads>(sum0);
  __syncthreads();
  sum1 = block_sum<Threads>(sum1);
  if (threadIdx.x == 0) {
    hidden0[batch * rank + row] = sum0 * __half2float(latent0[row]);
    hidden1[batch * rank + row] = sum1 * __half2float(latent1[row]);
  }
}

template <int Threads>
__global__ void t3_sparse_walb2_u_batched_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ t3,
    const uint8_t* __restrict__ positions,
    const uint8_t* __restrict__ sign_bits,
    const __half* __restrict__ alpha, const __half* __restrict__ beta,
    const float* __restrict__ hidden0, const float* __restrict__ hidden1,
    const uint8_t* __restrict__ u0, const uint8_t* __restrict__ u1,
    const __half* __restrict__ row0, const __half* __restrict__ row1,
    float* __restrict__ output,
    int rows, int columns, int rank, int groups) {
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  if (row >= rows) return;
  x += batch * columns;
  hidden0 += batch * rank;
  hidden1 += batch * rank;
  float base = 0.0f, wal0 = 0.0f, wal1 = 0.0f;
  const int t3_bytes = columns >> 2;
  for (int byte = threadIdx.x; byte < t3_bytes; byte += Threads) {
    const int group = byte >> 5;
    const int k = byte << 2;
    const uint8_t packed = t3[row * t3_bytes + byte];
    const float scale = __half2float(alpha[row * groups + group]);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const int code = static_cast<int>((packed >> (lane * 2)) & 3) - 1;
      base += scale * static_cast<float>(code) * __bfloat162float(x[k + lane]);
    }
  }
  for (int index = threadIdx.x; index < groups * 8; index += Threads) {
    const int group = index >> 3;
    const int slot = index & 7;
    const int position = positions[(row * groups + group) * 8 + slot];
    const uint8_t bits = sign_bits[row * groups + group];
    base += __half2float(beta[row * groups + group]) *
            (((bits >> slot) & 1) ? 1.0f : -1.0f) *
            __bfloat162float(x[group * 128 + position]);
  }
  const int u_bytes = rank >> 3;
  for (int byte = threadIdx.x; byte < u_bytes; byte += Threads) {
    const uint8_t packed0 = u0[row * u_bytes + byte];
    const uint8_t packed1 = u1[row * u_bytes + byte];
    const int k = byte << 3;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      wal0 += (((packed0 >> lane) & 1) ? 1.0f : -1.0f) * hidden0[k + lane];
      wal1 += (((packed1 >> lane) & 1) ? 1.0f : -1.0f) * hidden1[k + lane];
    }
  }
  block_sum_triple<Threads>(base, wal0, wal1);
  if (threadIdx.x == 0) {
    output[batch * rows + row] = base + wal0 * __half2float(row0[row]) +
                                        wal1 * __half2float(row1[row]);
  }
}

}  // namespace

torch::Tensor t3_sparse_batched(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    int64_t rows, int64_t columns, int64_t batch) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns && columns % 128 == 0, "invalid batch shape");
  auto output = torch::empty({batch, rows}, x.options().dtype(torch::kFloat32));
  dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch));
  t3_sparse_batched_kernel<128><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), output.data_ptr<float>(),
      static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(columns / 128));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> binary_pair_v_batched(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1,
    int64_t rank, int64_t columns, int64_t batch) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns && columns % 8 == 0, "invalid V batch shape");
  auto h0 = torch::empty({batch, rank}, x.options().dtype(torch::kFloat32));
  auto h1 = torch::empty({batch, rank}, x.options().dtype(torch::kFloat32));
  dim3 grid(static_cast<unsigned>(rank), static_cast<unsigned>(batch));
  binary_pair_v_batched_kernel<128><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), bits0.data_ptr<uint8_t>(),
      bits1.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(latent0.data_ptr()),
      reinterpret_cast<const __half*>(latent1.data_ptr()),
      reinterpret_cast<const __half*>(column0.data_ptr()),
      reinterpret_cast<const __half*>(column1.data_ptr()), h0.data_ptr<float>(),
      h1.data_ptr<float>(), static_cast<int>(rank), static_cast<int>(columns));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {h0, h1};
}

torch::Tensor t3_sparse_walb2_u_batched(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1,
    int64_t rows, int64_t columns, int64_t rank, int64_t batch) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns && hidden0.numel() == batch * rank,
              "invalid fused batch shape");
  auto output = torch::empty({batch, rows}, x.options().dtype(torch::kFloat32));
  dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch));
  t3_sparse_walb2_u_batched_kernel<128><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), hidden0.data_ptr<float>(),
      hidden1.data_ptr<float>(), u0.data_ptr<uint8_t>(), u1.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(row0.data_ptr()),
      reinterpret_cast<const __half*>(row1.data_ptr()), output.data_ptr<float>(),
      static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(rank),
      static_cast<int>(columns / 128));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("t3_sparse_batched", &t3_sparse_batched);
  module.def("binary_pair_v_batched", &binary_pair_v_batched);
  module.def("t3_sparse_walb2_u_batched", &t3_sparse_walb2_u_batched);
}
