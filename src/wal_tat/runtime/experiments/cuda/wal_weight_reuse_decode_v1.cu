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
__inline__ __device__ void block_sum_pair(float& a, float& b) {
  __shared__ float sa[Threads / 32];
  __shared__ float sb[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  a = warp_sum(a); b = warp_sum(b);
  if (lane == 0) { sa[warp] = a; sb[warp] = b; }
  __syncthreads();
  a = threadIdx.x < Threads / 32 ? sa[lane] : 0.0f;
  b = threadIdx.x < Threads / 32 ? sb[lane] : 0.0f;
  if (warp == 0) { a = warp_sum(a); b = warp_sum(b); }
  __syncthreads();
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
  __syncthreads();
}

#ifdef WAL_SINGLE_BARRIER_REDUCTION
template <int Batch, int Threads>
__inline__ __device__ void block_sum_batch(float (&values)[Batch]) {
  __shared__ float partial[Batch][Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    values[batch] = warp_sum(values[batch]);
    if (lane == 0) partial[batch][warp] = values[batch];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
      values[batch] = warp_sum(
          lane < Threads / 32 ? partial[batch][lane] : 0.0f);
    }
  }
}

template <int Batch, int Threads>
__inline__ __device__ void block_sum_pair_batch(
    float (&a)[Batch], float (&b)[Batch]) {
  __shared__ float partial_a[Batch][Threads / 32];
  __shared__ float partial_b[Batch][Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    a[batch] = warp_sum(a[batch]);
    b[batch] = warp_sum(b[batch]);
    if (lane == 0) {
      partial_a[batch][warp] = a[batch];
      partial_b[batch][warp] = b[batch];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
      a[batch] = warp_sum(
          lane < Threads / 32 ? partial_a[batch][lane] : 0.0f);
      b[batch] = warp_sum(
          lane < Threads / 32 ? partial_b[batch][lane] : 0.0f);
    }
  }
}

template <int Batch, int Threads>
__inline__ __device__ void block_sum_triple_batch(
    float (&a)[Batch], float (&b)[Batch], float (&c)[Batch]) {
  __shared__ float partial_a[Batch][Threads / 32];
  __shared__ float partial_b[Batch][Threads / 32];
  __shared__ float partial_c[Batch][Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    a[batch] = warp_sum(a[batch]);
    b[batch] = warp_sum(b[batch]);
    c[batch] = warp_sum(c[batch]);
    if (lane == 0) {
      partial_a[batch][warp] = a[batch];
      partial_b[batch][warp] = b[batch];
      partial_c[batch][warp] = c[batch];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
      a[batch] = warp_sum(
          lane < Threads / 32 ? partial_a[batch][lane] : 0.0f);
      b[batch] = warp_sum(
          lane < Threads / 32 ? partial_b[batch][lane] : 0.0f);
      c[batch] = warp_sum(
          lane < Threads / 32 ? partial_c[batch][lane] : 0.0f);
    }
  }
}
#endif

template <int Batch, int Threads>
__global__ void t3_sparse_reuse_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ t3,
    const uint8_t* __restrict__ positions,
    const uint8_t* __restrict__ sign_bits,
    const __half* __restrict__ alpha,
    const __half* __restrict__ beta,
    float* __restrict__ output,
    int rows, int columns, int groups) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int batch_base = blockIdx.y * Batch;
  float values[Batch];
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) values[batch] = 0.0f;
  const int bytes_per_row = columns >> 2;
#ifdef WAL_VECTOR_PACKED_LOADS
  const int words_per_row = bytes_per_row >> 2;
  const uint32_t* __restrict__ t3_words =
      reinterpret_cast<const uint32_t*>(t3);
  for (int word = threadIdx.x; word < words_per_row; word += Threads) {
    const int group = word >> 3;
    const int k = word << 4;
    const uint32_t packed = t3_words[row * words_per_row + word];
    const float scale = __half2float(alpha[row * groups + group]);
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const float weight = scale * static_cast<float>(
          static_cast<int>((packed >> (lane * 2)) & 3) - 1);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        values[batch] += weight * __bfloat162float(
            x[(batch_base + batch) * columns + k + lane]);
      }
    }
  }
#else
  for (int byte = threadIdx.x; byte < bytes_per_row; byte += Threads) {
    const int group = byte >> 5;
    const int k = byte << 2;
    const uint8_t packed = t3[row * bytes_per_row + byte];
    const float scale = __half2float(alpha[row * groups + group]);
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 4; lane += 2) {
      const float weight0 = scale * static_cast<float>(
          static_cast<int>((packed >> (lane * 2)) & 3) - 1);
      const float weight1 = scale * static_cast<float>(
          static_cast<int>((packed >> ((lane + 1) * 2)) & 3) - 1);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        const __nv_bfloat162 packed_input =
            *reinterpret_cast<const __nv_bfloat162*>(
                x + (batch_base + batch) * columns + k + lane);
        const float2 input = __bfloat1622float2(packed_input);
        values[batch] += weight0 * input.x;
        values[batch] += weight1 * input.y;
      }
    }
#else
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const float weight = scale *
          static_cast<float>(static_cast<int>((packed >> (lane * 2)) & 3) - 1);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        values[batch] += weight * __bfloat162float(
            x[(batch_base + batch) * columns + k + lane]);
      }
    }
#endif
  }
#endif
  for (int index = threadIdx.x; index < groups * 8; index += Threads) {
    const int group = index >> 3;
    const int slot = index & 7;
    const int position = positions[(row * groups + group) * 8 + slot];
    const uint8_t bits = sign_bits[row * groups + group];
    const float weight = __half2float(beta[row * groups + group]) *
                         (((bits >> slot) & 1) ? 1.0f : -1.0f);
    const int k = group * 128 + position;
#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
      values[batch] += weight * __bfloat162float(
          x[(batch_base + batch) * columns + k]);
    }
  }
#ifdef WAL_SINGLE_BARRIER_REDUCTION
  block_sum_batch<Batch, Threads>(values);
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    if (threadIdx.x == 0) {
      output[(batch_base + batch) * rows + row] = values[batch];
    }
  }
#else
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    float dummy = 0.0f;
    block_sum_pair<Threads>(values[batch], dummy);
    if (threadIdx.x == 0) {
      output[(batch_base + batch) * rows + row] = values[batch];
    }
  }
#endif
}

template <int Batch, int Threads>
__global__ void binary_pair_v_reuse_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ bits0, const uint8_t* __restrict__ bits1,
    const __half* __restrict__ latent0, const __half* __restrict__ latent1,
    const __half* __restrict__ column0, const __half* __restrict__ column1,
    float* __restrict__ hidden0, float* __restrict__ hidden1,
    int rank, int columns) {
  const int row = blockIdx.x;
  if (row >= rank) return;
  const int batch_base = blockIdx.y * Batch;
  float sum0[Batch], sum1[Batch];
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) { sum0[batch] = 0.0f; sum1[batch] = 0.0f; }
  const int bytes_per_row = columns >> 3;
#ifdef WAL_VECTOR_PACKED_LOADS
  const int words_per_row = bytes_per_row >> 2;
  const uint32_t* __restrict__ words0 =
      reinterpret_cast<const uint32_t*>(bits0);
  const uint32_t* __restrict__ words1 =
      reinterpret_cast<const uint32_t*>(bits1);
  for (int word = threadIdx.x; word < words_per_row; word += Threads) {
    const uint32_t packed0 = words0[row * words_per_row + word];
    const uint32_t packed1 = words1[row * words_per_row + word];
    const int k = word << 5;
#pragma unroll
    for (int lane = 0; lane < 32; ++lane) {
      const float weight0 = (((packed0 >> lane) & 1) ? 1.0f : -1.0f) *
                            __half2float(column0[k + lane]);
      const float weight1 = (((packed1 >> lane) & 1) ? 1.0f : -1.0f) *
                            __half2float(column1[k + lane]);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        const float input = __bfloat162float(
            x[(batch_base + batch) * columns + k + lane]);
        sum0[batch] += weight0 * input;
        sum1[batch] += weight1 * input;
      }
    }
  }
#else
  for (int byte = threadIdx.x; byte < bytes_per_row; byte += Threads) {
    const uint8_t packed0 = bits0[row * bytes_per_row + byte];
    const uint8_t packed1 = bits1[row * bytes_per_row + byte];
    const int k = byte << 3;
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 8; lane += 2) {
      const float2 columns0 = __half22float2(
          *reinterpret_cast<const __half2*>(column0 + k + lane));
      const float2 columns1 = __half22float2(
          *reinterpret_cast<const __half2*>(column1 + k + lane));
      const float weight00 = (((packed0 >> lane) & 1) ? 1.0f : -1.0f) * columns0.x;
      const float weight01 = (((packed0 >> (lane + 1)) & 1) ? 1.0f : -1.0f) * columns0.y;
      const float weight10 = (((packed1 >> lane) & 1) ? 1.0f : -1.0f) * columns1.x;
      const float weight11 = (((packed1 >> (lane + 1)) & 1) ? 1.0f : -1.0f) * columns1.y;
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        const __nv_bfloat162 packed_input =
            *reinterpret_cast<const __nv_bfloat162*>(
                x + (batch_base + batch) * columns + k + lane);
        const float2 input = __bfloat1622float2(packed_input);
        sum0[batch] += weight00 * input.x;
        sum0[batch] += weight01 * input.y;
        sum1[batch] += weight10 * input.x;
        sum1[batch] += weight11 * input.y;
      }
    }
#else
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const float weight0 = (((packed0 >> lane) & 1) ? 1.0f : -1.0f) *
                            __half2float(column0[k + lane]);
      const float weight1 = (((packed1 >> lane) & 1) ? 1.0f : -1.0f) *
                            __half2float(column1[k + lane]);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        const float input = __bfloat162float(
            x[(batch_base + batch) * columns + k + lane]);
        sum0[batch] += weight0 * input;
        sum1[batch] += weight1 * input;
      }
    }
#endif
  }
#endif
#ifdef WAL_SINGLE_BARRIER_REDUCTION
  block_sum_pair_batch<Batch, Threads>(sum0, sum1);
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    if (threadIdx.x == 0) {
      hidden0[(batch_base + batch) * rank + row] =
          sum0[batch] * __half2float(latent0[row]);
      hidden1[(batch_base + batch) * rank + row] =
          sum1[batch] * __half2float(latent1[row]);
    }
  }
#else
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    block_sum_pair<Threads>(sum0[batch], sum1[batch]);
    if (threadIdx.x == 0) {
      hidden0[(batch_base + batch) * rank + row] =
          sum0[batch] * __half2float(latent0[row]);
      hidden1[(batch_base + batch) * rank + row] =
          sum1[batch] * __half2float(latent1[row]);
    }
  }
#endif
}

template <int Batch, int Threads>
__global__ void t3_sparse_walb2_u_reuse_kernel(
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
  if (row >= rows) return;
  const int batch_base = blockIdx.y * Batch;
  float base[Batch], wal0[Batch], wal1[Batch];
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    base[batch] = 0.0f; wal0[batch] = 0.0f; wal1[batch] = 0.0f;
  }
  const int t3_bytes = columns >> 2;
#ifdef WAL_VECTOR_PACKED_LOADS
  const int t3_words_per_row = t3_bytes >> 2;
  const uint32_t* __restrict__ t3_words =
      reinterpret_cast<const uint32_t*>(t3);
  for (int word = threadIdx.x; word < t3_words_per_row; word += Threads) {
    const int group = word >> 3;
    const int k = word << 4;
    const uint32_t packed = t3_words[row * t3_words_per_row + word];
    const float scale = __half2float(alpha[row * groups + group]);
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const float weight = scale * static_cast<float>(
          static_cast<int>((packed >> (lane * 2)) & 3) - 1);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        base[batch] += weight * __bfloat162float(
            x[(batch_base + batch) * columns + k + lane]);
      }
    }
  }
#else
  for (int byte = threadIdx.x; byte < t3_bytes; byte += Threads) {
    const int group = byte >> 5;
    const int k = byte << 2;
    const uint8_t packed = t3[row * t3_bytes + byte];
    const float scale = __half2float(alpha[row * groups + group]);
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 4; lane += 2) {
      const float weight0 = scale * static_cast<float>(
          static_cast<int>((packed >> (lane * 2)) & 3) - 1);
      const float weight1 = scale * static_cast<float>(
          static_cast<int>((packed >> ((lane + 1) * 2)) & 3) - 1);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        const __nv_bfloat162 packed_input =
            *reinterpret_cast<const __nv_bfloat162*>(
                x + (batch_base + batch) * columns + k + lane);
        const float2 input = __bfloat1622float2(packed_input);
        base[batch] += weight0 * input.x;
        base[batch] += weight1 * input.y;
      }
    }
#else
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const float weight = scale *
          static_cast<float>(static_cast<int>((packed >> (lane * 2)) & 3) - 1);
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        base[batch] += weight * __bfloat162float(
            x[(batch_base + batch) * columns + k + lane]);
      }
    }
#endif
  }
#endif
  for (int index = threadIdx.x; index < groups * 8; index += Threads) {
    const int group = index >> 3;
    const int slot = index & 7;
    const int position = positions[(row * groups + group) * 8 + slot];
    const uint8_t bits = sign_bits[row * groups + group];
    const float weight = __half2float(beta[row * groups + group]) *
                         (((bits >> slot) & 1) ? 1.0f : -1.0f);
    const int k = group * 128 + position;
#pragma unroll
    for (int batch = 0; batch < Batch; ++batch) {
      base[batch] += weight * __bfloat162float(
          x[(batch_base + batch) * columns + k]);
    }
  }
  const int u_bytes = rank >> 3;
#ifdef WAL_VECTOR_PACKED_LOADS
  const int u_words_per_row = u_bytes >> 2;
  const uint32_t* __restrict__ u0_words =
      reinterpret_cast<const uint32_t*>(u0);
  const uint32_t* __restrict__ u1_words =
      reinterpret_cast<const uint32_t*>(u1);
  for (int word = threadIdx.x; word < u_words_per_row; word += Threads) {
    const uint32_t packed0 = u0_words[row * u_words_per_row + word];
    const uint32_t packed1 = u1_words[row * u_words_per_row + word];
    const int k = word << 5;
#pragma unroll
    for (int lane = 0; lane < 32; ++lane) {
      const float sign0 = ((packed0 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign1 = ((packed1 >> lane) & 1) ? 1.0f : -1.0f;
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        wal0[batch] += sign0 *
            hidden0[(batch_base + batch) * rank + k + lane];
        wal1[batch] += sign1 *
            hidden1[(batch_base + batch) * rank + k + lane];
      }
    }
  }
#else
  for (int byte = threadIdx.x; byte < u_bytes; byte += Threads) {
    const uint8_t packed0 = u0[row * u_bytes + byte];
    const uint8_t packed1 = u1[row * u_bytes + byte];
    const int k = byte << 3;
#ifdef WAL_VECTOR_ACTIVATION2
#pragma unroll
    for (int lane = 0; lane < 8; lane += 2) {
      const float sign00 = ((packed0 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign01 = ((packed0 >> (lane + 1)) & 1) ? 1.0f : -1.0f;
      const float sign10 = ((packed1 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign11 = ((packed1 >> (lane + 1)) & 1) ? 1.0f : -1.0f;
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        const float2 hidden_pair0 = *reinterpret_cast<const float2*>(
            hidden0 + (batch_base + batch) * rank + k + lane);
        const float2 hidden_pair1 = *reinterpret_cast<const float2*>(
            hidden1 + (batch_base + batch) * rank + k + lane);
        wal0[batch] += sign00 * hidden_pair0.x;
        wal0[batch] += sign01 * hidden_pair0.y;
        wal1[batch] += sign10 * hidden_pair1.x;
        wal1[batch] += sign11 * hidden_pair1.y;
      }
    }
#else
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const float sign0 = ((packed0 >> lane) & 1) ? 1.0f : -1.0f;
      const float sign1 = ((packed1 >> lane) & 1) ? 1.0f : -1.0f;
#pragma unroll
      for (int batch = 0; batch < Batch; ++batch) {
        wal0[batch] += sign0 *
            hidden0[(batch_base + batch) * rank + k + lane];
        wal1[batch] += sign1 *
            hidden1[(batch_base + batch) * rank + k + lane];
      }
    }
#endif
  }
#endif
#ifdef WAL_SINGLE_BARRIER_REDUCTION
  block_sum_triple_batch<Batch, Threads>(base, wal0, wal1);
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    if (threadIdx.x == 0) {
      output[(batch_base + batch) * rows + row] = base[batch] +
          wal0[batch] * __half2float(row0[row]) +
          wal1[batch] * __half2float(row1[row]);
    }
  }
#else
#pragma unroll
  for (int batch = 0; batch < Batch; ++batch) {
    block_sum_triple<Threads>(base[batch], wal0[batch], wal1[batch]);
    if (threadIdx.x == 0) {
      output[(batch_base + batch) * rows + row] = base[batch] +
          wal0[batch] * __half2float(row0[row]) +
          wal1[batch] * __half2float(row1[row]);
    }
  }
#endif
}

template <int Batch>
torch::Tensor launch_t3(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    int rows, int columns) {
  auto output = torch::empty({Batch, rows}, x.options().dtype(torch::kFloat32));
  t3_sparse_reuse_kernel<Batch, 128><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), output.data_ptr<float>(),
      rows, columns, columns / 128);
  return output;
}

template <int Batch>
std::vector<torch::Tensor> launch_v(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1, int rank, int columns) {
  auto h0 = torch::empty({Batch, rank}, x.options().dtype(torch::kFloat32));
  auto h1 = torch::empty({Batch, rank}, x.options().dtype(torch::kFloat32));
  binary_pair_v_reuse_kernel<Batch, 128><<<rank, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), bits0.data_ptr<uint8_t>(),
      bits1.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(latent0.data_ptr()),
      reinterpret_cast<const __half*>(latent1.data_ptr()),
      reinterpret_cast<const __half*>(column0.data_ptr()),
      reinterpret_cast<const __half*>(column1.data_ptr()), h0.data_ptr<float>(),
      h1.data_ptr<float>(), rank, columns);
  return {h0, h1};
}

template <int Batch>
torch::Tensor launch_fused(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1,
    int rows, int columns, int rank) {
  auto output = torch::empty({Batch, rows}, x.options().dtype(torch::kFloat32));
  t3_sparse_walb2_u_reuse_kernel<Batch, 128><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), hidden0.data_ptr<float>(),
      hidden1.data_ptr<float>(), u0.data_ptr<uint8_t>(), u1.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(row0.data_ptr()),
      reinterpret_cast<const __half*>(row1.data_ptr()), output.data_ptr<float>(),
      rows, columns, rank, columns / 128);
  return output;
}

template <int Tile>
torch::Tensor launch_t3_tiled(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    int rows, int columns, int total_batch) {
  auto output = torch::empty(
      {total_batch, rows}, x.options().dtype(torch::kFloat32));
  const dim3 grid(rows, total_batch / Tile);
  t3_sparse_reuse_kernel<Tile, 128><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), output.data_ptr<float>(),
      rows, columns, columns / 128);
  return output;
}

template <int Tile>
std::vector<torch::Tensor> launch_v_tiled(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1,
    int rank, int columns, int total_batch) {
  auto h0 = torch::empty(
      {total_batch, rank}, x.options().dtype(torch::kFloat32));
  auto h1 = torch::empty(
      {total_batch, rank}, x.options().dtype(torch::kFloat32));
  const dim3 grid(rank, total_batch / Tile);
  binary_pair_v_reuse_kernel<Tile, 128><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), bits0.data_ptr<uint8_t>(),
      bits1.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(latent0.data_ptr()),
      reinterpret_cast<const __half*>(latent1.data_ptr()),
      reinterpret_cast<const __half*>(column0.data_ptr()),
      reinterpret_cast<const __half*>(column1.data_ptr()), h0.data_ptr<float>(),
      h1.data_ptr<float>(), rank, columns);
  return {h0, h1};
}

template <int Tile>
torch::Tensor launch_fused_tiled(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1,
    int rows, int columns, int rank, int total_batch) {
  auto output = torch::empty(
      {total_batch, rows}, x.options().dtype(torch::kFloat32));
  const dim3 grid(rows, total_batch / Tile);
  t3_sparse_walb2_u_reuse_kernel<Tile, 128><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), hidden0.data_ptr<float>(),
      hidden1.data_ptr<float>(), u0.data_ptr<uint8_t>(), u1.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(row0.data_ptr()),
      reinterpret_cast<const __half*>(row1.data_ptr()), output.data_ptr<float>(),
      rows, columns, rank, columns / 128);
  return output;
}

template <int Threads>
torch::Tensor launch_t3_threads10(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    int rows, int columns) {
  auto output = torch::empty({10, rows}, x.options().dtype(torch::kFloat32));
  t3_sparse_reuse_kernel<10, Threads><<<rows, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), output.data_ptr<float>(),
      rows, columns, columns / 128);
  return output;
}

template <int Threads>
std::vector<torch::Tensor> launch_v_threads10(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1,
    int rank, int columns) {
  auto h0 = torch::empty({10, rank}, x.options().dtype(torch::kFloat32));
  auto h1 = torch::empty({10, rank}, x.options().dtype(torch::kFloat32));
  binary_pair_v_reuse_kernel<10, Threads><<<rank, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), bits0.data_ptr<uint8_t>(),
      bits1.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(latent0.data_ptr()),
      reinterpret_cast<const __half*>(latent1.data_ptr()),
      reinterpret_cast<const __half*>(column0.data_ptr()),
      reinterpret_cast<const __half*>(column1.data_ptr()), h0.data_ptr<float>(),
      h1.data_ptr<float>(), rank, columns);
  return {h0, h1};
}

template <int Threads>
torch::Tensor launch_fused_threads10(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1,
    int rows, int columns, int rank) {
  auto output = torch::empty({10, rows}, x.options().dtype(torch::kFloat32));
  t3_sparse_walb2_u_reuse_kernel<10, Threads><<<rows, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), t3.data_ptr<uint8_t>(),
      positions.data_ptr<uint8_t>(), sign_bits.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(alpha.data_ptr()),
      reinterpret_cast<const __half*>(beta.data_ptr()), hidden0.data_ptr<float>(),
      hidden1.data_ptr<float>(), u0.data_ptr<uint8_t>(), u1.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(row0.data_ptr()),
      reinterpret_cast<const __half*>(row1.data_ptr()), output.data_ptr<float>(),
      rows, columns, rank, columns / 128);
  return output;
}

#define DISPATCH_BATCH(FUNCTION, ...) \
  switch (batch) { \
    case 2: return FUNCTION<2>(__VA_ARGS__); case 3: return FUNCTION<3>(__VA_ARGS__); \
    case 4: return FUNCTION<4>(__VA_ARGS__); case 5: return FUNCTION<5>(__VA_ARGS__); \
    case 6: return FUNCTION<6>(__VA_ARGS__); case 7: return FUNCTION<7>(__VA_ARGS__); \
    case 8: return FUNCTION<8>(__VA_ARGS__); case 9: return FUNCTION<9>(__VA_ARGS__); \
    case 10: return FUNCTION<10>(__VA_ARGS__); case 11: return FUNCTION<11>(__VA_ARGS__); \
    case 12: return FUNCTION<12>(__VA_ARGS__); case 13: return FUNCTION<13>(__VA_ARGS__); \
    case 14: return FUNCTION<14>(__VA_ARGS__); case 15: return FUNCTION<15>(__VA_ARGS__); \
    case 16: return FUNCTION<16>(__VA_ARGS__); \
    default: TORCH_CHECK(false, "batch must be in [2,16]"); \
  }

}  // namespace

torch::Tensor t3_sparse_reuse(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    int64_t rows, int64_t columns, int64_t batch) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns, "invalid input shape");
  DISPATCH_BATCH(launch_t3, x, t3, positions, sign_bits, alpha, beta,
                 static_cast<int>(rows), static_cast<int>(columns));
}

std::vector<torch::Tensor> binary_pair_v_reuse(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1,
    int64_t rank, int64_t columns, int64_t batch) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns, "invalid input shape");
  DISPATCH_BATCH(launch_v, x, bits0, bits1, latent0, latent1, column0, column1,
                 static_cast<int>(rank), static_cast<int>(columns));
}

torch::Tensor t3_sparse_walb2_u_reuse(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1,
    int64_t rows, int64_t columns, int64_t rank, int64_t batch) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns && hidden0.numel() == batch * rank,
              "invalid input/hidden shape");
  DISPATCH_BATCH(launch_fused, x, t3, positions, sign_bits, alpha, beta,
                 hidden0, hidden1, u0, u1, row0, row1,
                 static_cast<int>(rows), static_cast<int>(columns),
                 static_cast<int>(rank));
}

torch::Tensor t3_sparse_tiled(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    int64_t rows, int64_t columns, int64_t batch, int64_t tile) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns && batch % tile == 0,
              "invalid input shape or tile");
  switch (tile) {
    case 1: return launch_t3_tiled<1>(x, t3, positions, sign_bits, alpha, beta,
        rows, columns, batch);
    case 2: return launch_t3_tiled<2>(x, t3, positions, sign_bits, alpha, beta,
        rows, columns, batch);
    case 5: return launch_t3_tiled<5>(x, t3, positions, sign_bits, alpha, beta,
        rows, columns, batch);
    case 10: return launch_t3_tiled<10>(x, t3, positions, sign_bits, alpha, beta,
        rows, columns, batch);
    default: TORCH_CHECK(false, "tile must be one of 1,2,5,10");
  }
}

std::vector<torch::Tensor> binary_pair_v_tiled(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1,
    int64_t rank, int64_t columns, int64_t batch, int64_t tile) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns && batch % tile == 0,
              "invalid input shape or tile");
  switch (tile) {
    case 1: return launch_v_tiled<1>(x, bits0, bits1, latent0, latent1,
        column0, column1, rank, columns, batch);
    case 2: return launch_v_tiled<2>(x, bits0, bits1, latent0, latent1,
        column0, column1, rank, columns, batch);
    case 5: return launch_v_tiled<5>(x, bits0, bits1, latent0, latent1,
        column0, column1, rank, columns, batch);
    case 10: return launch_v_tiled<10>(x, bits0, bits1, latent0, latent1,
        column0, column1, rank, columns, batch);
    default: TORCH_CHECK(false, "tile must be one of 1,2,5,10");
  }
}

torch::Tensor t3_sparse_walb2_u_tiled(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1,
    int64_t rows, int64_t columns, int64_t rank,
    int64_t batch, int64_t tile) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "BF16 CUDA x required");
  TORCH_CHECK(x.numel() == batch * columns && hidden0.numel() == batch * rank &&
              batch % tile == 0, "invalid input/hidden shape or tile");
  switch (tile) {
    case 1: return launch_fused_tiled<1>(x, t3, positions, sign_bits, alpha, beta,
        hidden0, hidden1, u0, u1, row0, row1, rows, columns, rank, batch);
    case 2: return launch_fused_tiled<2>(x, t3, positions, sign_bits, alpha, beta,
        hidden0, hidden1, u0, u1, row0, row1, rows, columns, rank, batch);
    case 5: return launch_fused_tiled<5>(x, t3, positions, sign_bits, alpha, beta,
        hidden0, hidden1, u0, u1, row0, row1, rows, columns, rank, batch);
    case 10: return launch_fused_tiled<10>(x, t3, positions, sign_bits, alpha, beta,
        hidden0, hidden1, u0, u1, row0, row1, rows, columns, rank, batch);
    default: TORCH_CHECK(false, "tile must be one of 1,2,5,10");
  }
}

torch::Tensor t3_sparse_threads10(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    int64_t rows, int64_t columns, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 &&
              x.numel() == 10 * columns, "batch-10 BF16 CUDA x required");
  switch (threads) {
    case 64: return launch_t3_threads10<64>(
        x, t3, positions, sign_bits, alpha, beta, rows, columns);
    case 128: return launch_t3_threads10<128>(
        x, t3, positions, sign_bits, alpha, beta, rows, columns);
    case 256: return launch_t3_threads10<256>(
        x, t3, positions, sign_bits, alpha, beta, rows, columns);
    default: TORCH_CHECK(false, "threads must be 64, 128 or 256");
  }
}

std::vector<torch::Tensor> binary_pair_v_threads10(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    torch::Tensor latent0, torch::Tensor latent1,
    torch::Tensor column0, torch::Tensor column1,
    int64_t rank, int64_t columns, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 &&
              x.numel() == 10 * columns, "batch-10 BF16 CUDA x required");
  switch (threads) {
    case 64: return launch_v_threads10<64>(
        x, bits0, bits1, latent0, latent1, column0, column1, rank, columns);
    case 128: return launch_v_threads10<128>(
        x, bits0, bits1, latent0, latent1, column0, column1, rank, columns);
    case 256: return launch_v_threads10<256>(
        x, bits0, bits1, latent0, latent1, column0, column1, rank, columns);
    default: TORCH_CHECK(false, "threads must be 64, 128 or 256");
  }
}

torch::Tensor t3_sparse_walb2_u_threads10(
    torch::Tensor x, torch::Tensor t3, torch::Tensor positions,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1,
    int64_t rows, int64_t columns, int64_t rank, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 &&
              x.numel() == 10 * columns && hidden0.numel() == 10 * rank,
              "batch-10 BF16 CUDA x and hidden required");
  switch (threads) {
    case 64: return launch_fused_threads10<64>(
        x, t3, positions, sign_bits, alpha, beta, hidden0, hidden1,
        u0, u1, row0, row1, rows, columns, rank);
    case 128: return launch_fused_threads10<128>(
        x, t3, positions, sign_bits, alpha, beta, hidden0, hidden1,
        u0, u1, row0, row1, rows, columns, rank);
    case 256: return launch_fused_threads10<256>(
        x, t3, positions, sign_bits, alpha, beta, hidden0, hidden1,
        u0, u1, row0, row1, rows, columns, rank);
    default: TORCH_CHECK(false, "threads must be 64, 128 or 256");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("t3_sparse_reuse", &t3_sparse_reuse);
  module.def("binary_pair_v_reuse", &binary_pair_v_reuse);
  module.def("t3_sparse_walb2_u_reuse", &t3_sparse_walb2_u_reuse);
  module.def("t3_sparse_tiled", &t3_sparse_tiled);
  module.def("binary_pair_v_tiled", &binary_pair_v_tiled);
  module.def("t3_sparse_walb2_u_tiled", &t3_sparse_walb2_u_tiled);
  module.def("t3_sparse_threads10", &t3_sparse_threads10);
  module.def("binary_pair_v_threads10", &binary_pair_v_threads10);
  module.def("t3_sparse_walb2_u_threads10", &t3_sparse_walb2_u_threads10);
}
