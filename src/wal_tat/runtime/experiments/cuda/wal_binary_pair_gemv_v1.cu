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
__inline__ __device__ void block_sum_pair(float& value0, float& value1) {
  __shared__ float sums0[Threads / 32];
  __shared__ float sums1[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value0 = warp_sum(value0);
  value1 = warp_sum(value1);
  if (lane == 0) {
    sums0[warp] = value0;
    sums1[warp] = value1;
  }
  __syncthreads();
  value0 = threadIdx.x < Threads / 32 ? sums0[lane] : 0.0f;
  value1 = threadIdx.x < Threads / 32 ? sums1[lane] : 0.0f;
  if (warp == 0) {
    value0 = warp_sum(value0);
    value1 = warp_sum(value1);
  }
}

template <int Threads>
__inline__ __device__ void block_sum_triple(
    float& value0, float& value1, float& value2) {
  __shared__ float sums0[Threads / 32];
  __shared__ float sums1[Threads / 32];
  __shared__ float sums2[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value0 = warp_sum(value0);
  value1 = warp_sum(value1);
  value2 = warp_sum(value2);
  if (lane == 0) {
    sums0[warp] = value0;
    sums1[warp] = value1;
    sums2[warp] = value2;
  }
  __syncthreads();
  value0 = threadIdx.x < Threads / 32 ? sums0[lane] : 0.0f;
  value1 = threadIdx.x < Threads / 32 ? sums1[lane] : 0.0f;
  value2 = threadIdx.x < Threads / 32 ? sums2[lane] : 0.0f;
  if (warp == 0) {
    value0 = warp_sum(value0);
    value1 = warp_sum(value1);
    value2 = warp_sum(value2);
  }
}

template <int Threads>
__global__ void binary_pair_v_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ bits0,
    const uint8_t* __restrict__ bits1,
    const __half* __restrict__ latent0,
    const __half* __restrict__ latent1,
    const __half* __restrict__ column0,
    const __half* __restrict__ column1,
    float* __restrict__ hidden0,
    float* __restrict__ hidden1,
    int rank,
    int columns) {
  const int row = blockIdx.x;
  if (row >= rank) return;
  const int bytes_per_row = columns >> 3;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  for (int byte = threadIdx.x; byte < bytes_per_row; byte += Threads) {
    const uint8_t packed0 = bits0[row * bytes_per_row + byte];
    const uint8_t packed1 = bits1[row * bytes_per_row + byte];
    const int k = byte << 3;
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 8; lane += 2) {
      const __nv_bfloat162 packed_input =
          *reinterpret_cast<const __nv_bfloat162*>(x + k + lane);
      const float2 input = __bfloat1622float2(packed_input);
      const float2 columns0 = __half22float2(
          *reinterpret_cast<const __half2*>(column0 + k + lane));
      const float2 columns1 = __half22float2(
          *reinterpret_cast<const __half2*>(column1 + k + lane));
      const float sign00 = ((packed0 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign01 = ((packed0 >> (lane + 1)) & 1) ? 1.0f : -1.0f;
      const float sign10 = ((packed1 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign11 = ((packed1 >> (lane + 1)) & 1) ? 1.0f : -1.0f;
      sum0 += sign00 * input.x * columns0.x;
      sum0 += sign01 * input.y * columns0.y;
      sum1 += sign10 * input.x * columns1.x;
      sum1 += sign11 * input.y * columns1.y;
    }
#else
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const float input = __bfloat162float(x[k + lane]);
      const float sign0 = ((packed0 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign1 = ((packed1 >> lane) & 1) ? 1.0f : -1.0f;
      sum0 += sign0 * input * __half2float(column0[k + lane]);
      sum1 += sign1 * input * __half2float(column1[k + lane]);
    }
#endif
  }
  block_sum_pair<Threads>(sum0, sum1);
  if (threadIdx.x == 0) {
    hidden0[row] = sum0 * __half2float(latent0[row]);
    hidden1[row] = sum1 * __half2float(latent1[row]);
  }
}

template <int Threads>
__global__ void binary_pair_u_accumulate_kernel(
    const float* __restrict__ hidden0,
    const float* __restrict__ hidden1,
    const uint8_t* __restrict__ bits0,
    const uint8_t* __restrict__ bits1,
    const __half* __restrict__ row0,
    const __half* __restrict__ row1,
    float* __restrict__ output,
    int rows,
    int rank) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int bytes_per_row = rank >> 3;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  for (int byte = threadIdx.x; byte < bytes_per_row; byte += Threads) {
    const uint8_t packed0 = bits0[row * bytes_per_row + byte];
    const uint8_t packed1 = bits1[row * bytes_per_row + byte];
    const int k = byte << 3;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const float sign0 = ((packed0 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign1 = ((packed1 >> lane) & 1) ? 1.0f : -1.0f;
      sum0 += sign0 * hidden0[k + lane];
      sum1 += sign1 * hidden1[k + lane];
    }
  }
  block_sum_pair<Threads>(sum0, sum1);
  if (threadIdx.x == 0) {
    output[row] += sum0 * __half2float(row0[row]) +
                   sum1 * __half2float(row1[row]);
  }
}

template <int Threads>
__global__ void t3_sparse_walb2_u_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ t3,
    const uint8_t* __restrict__ positions,
    const uint8_t* __restrict__ sign_bits,
    const __half* __restrict__ alpha,
    const __half* __restrict__ beta,
    const float* __restrict__ hidden0,
    const float* __restrict__ hidden1,
    const uint8_t* __restrict__ u0,
    const uint8_t* __restrict__ u1,
    const __half* __restrict__ row0,
    const __half* __restrict__ row1,
    float* __restrict__ output,
    int rows,
    int columns,
    int rank,
    int groups) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int t3_bytes_per_row = columns >> 2;
  float base = 0.0f;
  float wal0 = 0.0f;
  float wal1 = 0.0f;

  for (int byte = threadIdx.x; byte < t3_bytes_per_row; byte += Threads) {
    const int group = byte >> 5;
    const int k = byte << 2;
    const uint8_t packed = t3[row * t3_bytes_per_row + byte];
    const float scale = __half2float(alpha[row * groups + group]);
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 4; lane += 2) {
      const int code0 = static_cast<int>((packed >> (lane * 2)) & 3) - 1;
      const int code1 = static_cast<int>((packed >> ((lane + 1) * 2)) & 3) - 1;
      const __nv_bfloat162 packed_input =
          *reinterpret_cast<const __nv_bfloat162*>(x + k + lane);
      const float2 input = __bfloat1622float2(packed_input);
      base += scale * static_cast<float>(code0) * input.x;
      base += scale * static_cast<float>(code1) * input.y;
    }
#else
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const int code = static_cast<int>((packed >> (lane * 2)) & 3) - 1;
      base += scale * static_cast<float>(code) * __bfloat162float(x[k + lane]);
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
    base += __half2float(beta[row * groups + group]) * sign *
            __bfloat162float(x[group * 128 + position]);
  }

  const int u_bytes_per_row = rank >> 3;
  for (int byte = threadIdx.x; byte < u_bytes_per_row; byte += Threads) {
    const uint8_t packed0 = u0[row * u_bytes_per_row + byte];
    const uint8_t packed1 = u1[row * u_bytes_per_row + byte];
    const int k = byte << 3;
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 8; lane += 2) {
      const float2 hidden_pair0 =
          *reinterpret_cast<const float2*>(hidden0 + k + lane);
      const float2 hidden_pair1 =
          *reinterpret_cast<const float2*>(hidden1 + k + lane);
      wal0 += (((packed0 >> lane) & 1) ? 1.0f : -1.0f) * hidden_pair0.x;
      wal0 += (((packed0 >> (lane + 1)) & 1) ? 1.0f : -1.0f) * hidden_pair0.y;
      wal1 += (((packed1 >> lane) & 1) ? 1.0f : -1.0f) * hidden_pair1.x;
      wal1 += (((packed1 >> (lane + 1)) & 1) ? 1.0f : -1.0f) * hidden_pair1.y;
    }
#else
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      wal0 += (((packed0 >> lane) & 1) ? 1.0f : -1.0f) * hidden0[k + lane];
      wal1 += (((packed1 >> lane) & 1) ? 1.0f : -1.0f) * hidden1[k + lane];
    }
#endif
  }
  block_sum_triple<Threads>(base, wal0, wal1);
  if (threadIdx.x == 0) {
    output[row] = base + wal0 * __half2float(row0[row]) +
                         wal1 * __half2float(row1[row]);
  }
}

}  // namespace

std::vector<torch::Tensor> binary_pair_v_cuda(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1,
    int64_t rank, int64_t columns, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && bits0.is_cuda() && bits1.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(bits0.scalar_type() == torch::kUInt8 && bits1.scalar_type() == torch::kUInt8,
              "bits must be uint8");
  TORCH_CHECK(latent0.scalar_type() == torch::kFloat16 && latent1.scalar_type() == torch::kFloat16,
              "latent scales must be float16");
  TORCH_CHECK(column0.scalar_type() == torch::kFloat16 && column1.scalar_type() == torch::kFloat16,
              "column scales must be float16");
  TORCH_CHECK(columns % 8 == 0 && x.numel() == columns, "invalid V shape");
  auto hidden0 = torch::empty({rank}, x.options().dtype(torch::kFloat32));
  auto hidden1 = torch::empty({rank}, x.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 128) {
    binary_pair_v_kernel<128><<<rank, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        bits0.data_ptr<uint8_t>(), bits1.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(latent0.data_ptr()),
        reinterpret_cast<const __half*>(latent1.data_ptr()),
        reinterpret_cast<const __half*>(column0.data_ptr()),
        reinterpret_cast<const __half*>(column1.data_ptr()),
        hidden0.data_ptr<float>(), hidden1.data_ptr<float>(),
        static_cast<int>(rank), static_cast<int>(columns));
  } else {
    TORCH_CHECK(false, "threads must be 128");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {hidden0, hidden1};
}

torch::Tensor binary_pair_u_accumulate_cuda(
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor row0, torch::Tensor row1,
    torch::Tensor output, int64_t rows, int64_t rank, int64_t threads) {
  TORCH_CHECK(hidden0.is_cuda() && hidden1.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(hidden0.scalar_type() == torch::kFloat32 && hidden1.scalar_type() == torch::kFloat32,
              "hidden tensors must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(bits0.scalar_type() == torch::kUInt8 && bits1.scalar_type() == torch::kUInt8,
              "bits must be uint8");
  TORCH_CHECK(row0.scalar_type() == torch::kFloat16 && row1.scalar_type() == torch::kFloat16,
              "row scales must be float16");
  TORCH_CHECK(rank % 8 == 0 && hidden0.numel() == rank && hidden1.numel() == rank,
              "invalid U shape");
  TORCH_CHECK(output.numel() == rows, "output shape mismatch");
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 128) {
    binary_pair_u_accumulate_kernel<128><<<rows, 128, 0, stream>>>(
        hidden0.data_ptr<float>(), hidden1.data_ptr<float>(),
        bits0.data_ptr<uint8_t>(), bits1.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(row0.data_ptr()),
        reinterpret_cast<const __half*>(row1.data_ptr()),
        output.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(rank));
  } else {
    TORCH_CHECK(false, "threads must be 128");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor t3_sparse_walb2_u_cuda(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1,
    torch::Tensor row0, torch::Tensor row1,
    int64_t rows, int64_t columns, int64_t rank, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && t3.is_cuda() && hidden0.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(t3.scalar_type() == torch::kUInt8 && positions.scalar_type() == torch::kUInt8 &&
              sign_bits.scalar_type() == torch::kUInt8, "base codes must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat16 && beta.scalar_type() == torch::kFloat16,
              "base scales must be float16");
  TORCH_CHECK(hidden0.scalar_type() == torch::kFloat32 && hidden1.scalar_type() == torch::kFloat32,
              "hidden tensors must be float32");
  TORCH_CHECK(u0.scalar_type() == torch::kUInt8 && u1.scalar_type() == torch::kUInt8,
              "U bits must be uint8");
  TORCH_CHECK(row0.scalar_type() == torch::kFloat16 && row1.scalar_type() == torch::kFloat16,
              "row scales must be float16");
  TORCH_CHECK(columns % 128 == 0 && rank % 8 == 0, "invalid fused dimensions");
  TORCH_CHECK(x.numel() == columns && hidden0.numel() == rank && hidden1.numel() == rank,
              "fused activation shape mismatch");
  auto output = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 128) {
    t3_sparse_walb2_u_kernel<128><<<rows, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        t3.data_ptr<uint8_t>(), positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(alpha.data_ptr()),
        reinterpret_cast<const __half*>(beta.data_ptr()),
        hidden0.data_ptr<float>(), hidden1.data_ptr<float>(),
        u0.data_ptr<uint8_t>(), u1.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(row0.data_ptr()),
        reinterpret_cast<const __half*>(row1.data_ptr()), output.data_ptr<float>(),
        static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(rank),
        static_cast<int>(columns / 128));
  } else {
    TORCH_CHECK(false, "threads must be 128");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("binary_pair_v", &binary_pair_v_cuda, "WALB2 paired V GEMV");
  module.def("binary_pair_u_accumulate", &binary_pair_u_accumulate_cuda,
             "WALB2 paired U GEMV with base accumulation");
  module.def("t3_sparse_walb2_u", &t3_sparse_walb2_u_cuda,
             "Fused T3+sparse-k8 and paired WALB2 U GEMV");
}
