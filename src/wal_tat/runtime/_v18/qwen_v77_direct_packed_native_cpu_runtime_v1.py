"""Correctness-first native CPU runtime for the V77 direct-packed compute ABI."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from pathlib import Path
import resource
import time

import torch
from torch.utils.cpp_extension import load_inline

import qwen_v77_packed_reference_runtime as ref
from wal_tat.runtime.cache import (
    load_hardware_base_cache,
    load_packed_overlay,
)


CPP = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

struct TernaryMaskLUT {
  uint8_t positive[256];
  uint8_t negative[256];
  TernaryMaskLUT() {
    for (int byte=0;byte<256;++byte) {
      unsigned pos=0,neg=0;
      for (int lane=0;lane<4;++lane) {
        const int symbol=(byte>>(2*lane))&3;
        pos|=unsigned(symbol==2)<<lane;
        neg|=unsigned(symbol==0)<<lane;
      }
      positive[byte]=uint8_t(pos); negative[byte]=uint8_t(neg);
    }
  }
};
static const TernaryMaskLUT T3_MASKS;

#if defined(__ARM_NEON)
static inline float32x4_t wal_sign4(const uint8_t* bits, int64_t base) {
  int32_t signs[4];
  for (int lane=0; lane<4; ++lane)
    signs[lane]=int((bits[(base+lane)>>3]>>((base+lane)&7))&1)*2-1;
  return vcvtq_f32_s32(vld1q_s32(signs));
}

static inline float wal_neon_ternary_dot(
    const float* x, const uint8_t* packed, int width) {
  float32x4_t acc=vdupq_n_f32(0.0f);
  for(int k=0;k<width;k+=4){
    const uint8_t byte=packed[k>>2];
    int32_t codes[4]={
      int(byte&3)-1, int((byte>>2)&3)-1,
      int((byte>>4)&3)-1, int((byte>>6)&3)-1};
    acc=vmlaq_f32(acc,vld1q_f32(x+k),vcvtq_f32_s32(vld1q_s32(codes)));
  }
  return vaddvq_f32(acc);
}

static inline float wal_neon_binary_dot(
    const float* x, const uint8_t* bits, int64_t base, int width,
    const at::Half* column) {
  float32x4_t acc=vdupq_n_f32(0.0f);
  for(int k=0;k<width;k+=4){
    float32x4_t value=vld1q_f32(x+k);
    if(column){
      float scales[4]={float(column[k]),float(column[k+1]),
                       float(column[k+2]),float(column[k+3])};
      value=vmulq_f32(value,vld1q_f32(scales));
    }
    acc=vmlaq_f32(acc,value,wal_sign4(bits,base+k));
  }
  return vaddvq_f32(acc);
}
#endif

torch::Tensor t3_sparse_packed_cpu(
    torch::Tensor x, torch::Tensor t3, torch::Tensor pos,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta) {
  x=x.contiguous().to(torch::kFloat32); t3=t3.contiguous(); pos=pos.contiguous();
  sign_bits=sign_bits.contiguous(); alpha=alpha.contiguous(); beta=beta.contiguous();
  const int64_t M=x.size(0), N=t3.size(0), K=t3.size(1)*4, G=K/128, BPR=K/4;
  auto out=torch::empty({M,N}, torch::TensorOptions().dtype(torch::kFloat32));
  const float* X=x.data_ptr<float>(); const uint8_t* T=t3.data_ptr<uint8_t>();
  const uint8_t* P=pos.data_ptr<uint8_t>(); const uint8_t* SB=sign_bits.data_ptr<uint8_t>();
  const at::Half* A=alpha.data_ptr<at::Half>(); const at::Half* E=beta.data_ptr<at::Half>();
  float* O=out.data_ptr<float>();
  at::parallel_for(0,M*N,16,[&](int64_t begin,int64_t end){
    for(int64_t z=begin;z<end;++z){
      const int64_t m=z/N,n=z%N; float acc=0.0f;
      for(int64_t g=0;g<G;++g){
        const float* xv=X+m*K+g*128; float dense=0.0f;
        const uint8_t* row=T+n*BPR+g*32;
#if defined(__AVX512F__)
        __m512 densev=_mm512_setzero_ps();
        for(int k0=0;k0<128;k0+=16){
          unsigned positive=0,negative=0;
          for(int lane=0;lane<4;++lane){
            const uint8_t packed=row[(k0>>2)+lane];
            positive |= unsigned(T3_MASKS.positive[packed])<<(lane*4);
            negative |= unsigned(T3_MASKS.negative[packed])<<(lane*4);
          }
          const __m512 value=_mm512_loadu_ps(xv+k0);
          densev=_mm512_mask_add_ps(densev,(__mmask16)positive,densev,value);
          densev=_mm512_mask_sub_ps(densev,(__mmask16)negative,densev,value);
        }
        dense=_mm512_reduce_add_ps(densev);
#elif defined(__ARM_NEON)
        dense=wal_neon_ternary_dot(xv,row,128);
#else
        #pragma omp simd reduction(+:dense)
        for(int k=0;k<128;++k){
          const int code=int((row[k>>2] >> ((k&3)*2)) & 3)-1;
          dense += xv[k]*float(code);
        }
#endif
        const int64_t q=(n*G+g)*8; const uint8_t signs=SB[n*G+g];
        float sparse=0.0f;
        for(int j=0;j<8;++j)
          sparse += xv[P[q+j]]*float(((signs>>j)&1)*2-1);
        acc += dense*float(A[n*G+g])+sparse*float(E[n*G+g]);
      }
      O[z]=acc;
    }
  }); return out;
}

torch::Tensor binary_packed_cpu(
    torch::Tensor x, torch::Tensor bits, int64_t N, int64_t K,
    torch::Tensor row, torch::Tensor column) {
  x=x.contiguous().to(torch::kFloat32); bits=bits.contiguous();
  row=row.contiguous(); column=column.contiguous();
  const int64_t M=x.size(0); const bool has_col=column.numel()==K;
  auto out=torch::empty({M,N},torch::TensorOptions().dtype(torch::kFloat32));
  const float* X=x.data_ptr<float>(); const uint8_t* B=bits.data_ptr<uint8_t>();
  const at::Half* R=row.data_ptr<at::Half>();
  const at::Half* C=has_col?column.data_ptr<at::Half>():nullptr; float* O=out.data_ptr<float>();
  at::parallel_for(0,M*N,16,[&](int64_t begin,int64_t end){
    for(int64_t z=begin;z<end;++z){
      const int64_t m=z/N,n=z%N; float acc=0.0f; const int64_t base=n*K;
#if defined(__AVX512F__)
      __m512 accv=_mm512_setzero_ps();
      for(int64_t k=0;k<K;k+=16){
        __m512 value=_mm512_loadu_ps(X+m*K+k);
        if(has_col){
          const __m256i halfs=_mm256_loadu_si256((const __m256i*)(C+k));
          value=_mm512_mul_ps(value,_mm512_cvtph_ps(halfs));
        }
        const int64_t bit=base+k;
        const uint8_t* sign_ptr=B+(bit>>3);
        const __mmask16 mask=(__mmask16)(unsigned(sign_ptr[0])|(unsigned(sign_ptr[1])<<8));
        const __m512 negative=_mm512_sub_ps(_mm512_setzero_ps(),value);
        const __m512 signed_value=_mm512_mask_mov_ps(negative,mask,value);
        accv=_mm512_add_ps(accv,signed_value);
      }
      acc=_mm512_reduce_add_ps(accv);
#elif defined(__ARM_NEON)
      acc=wal_neon_binary_dot(X+m*K,B,base,K,has_col?C:nullptr);
#else
      if(has_col){
        #pragma omp simd reduction(+:acc)
        for(int64_t k=0;k<K;++k){
          const int code=int((B[(base+k)>>3]>>((base+k)&7))&1)*2-1;
          acc += X[m*K+k]*float(C[k])*float(code);
        }
      } else {
        #pragma omp simd reduction(+:acc)
        for(int64_t k=0;k<K;++k){
          const int code=int((B[(base+k)>>3]>>((base+k)&7))&1)*2-1;
          acc += X[m*K+k]*float(code);
        }
      }
#endif
      O[z]=acc*float(R[n]);
    }
  }); return out;
}

std::vector<torch::Tensor> binary_pair_packed_cpu(
    torch::Tensor x, torch::Tensor bits0, torch::Tensor bits1,
    int64_t N, int64_t K, torch::Tensor row0, torch::Tensor row1,
    torch::Tensor column0, torch::Tensor column1) {
  x=x.contiguous().to(torch::kFloat32); bits0=bits0.contiguous(); bits1=bits1.contiguous();
  row0=row0.contiguous(); row1=row1.contiguous();
  column0=column0.contiguous(); column1=column1.contiguous();
  const int64_t M=x.size(0);
  auto out0=torch::empty({M,N},torch::TensorOptions().dtype(torch::kFloat32));
  auto out1=torch::empty({M,N},torch::TensorOptions().dtype(torch::kFloat32));
  const float* X=x.data_ptr<float>();
  const uint8_t* B0=bits0.data_ptr<uint8_t>(); const uint8_t* B1=bits1.data_ptr<uint8_t>();
  const at::Half* R0=row0.data_ptr<at::Half>(); const at::Half* R1=row1.data_ptr<at::Half>();
  const at::Half* C0=column0.data_ptr<at::Half>(); const at::Half* C1=column1.data_ptr<at::Half>();
  float* O0=out0.data_ptr<float>(); float* O1=out1.data_ptr<float>();
  at::parallel_for(0,M*N,16,[&](int64_t begin,int64_t end){
    for(int64_t z=begin;z<end;++z){
      const int64_t m=z/N,n=z%N,base=n*K;
#if defined(__AVX512F__)
      __m512 acc0=_mm512_setzero_ps(),acc1=_mm512_setzero_ps();
      for(int64_t k=0;k<K;k+=16){
        const __m512 value=_mm512_loadu_ps(X+m*K+k);
        const __m256i halves0=_mm256_loadu_si256((const __m256i*)(C0+k));
        const __m256i halves1=_mm256_loadu_si256((const __m256i*)(C1+k));
        const __m512 scaled0=_mm512_mul_ps(value,_mm512_cvtph_ps(halves0));
        const __m512 scaled1=_mm512_mul_ps(value,_mm512_cvtph_ps(halves1));
        const int64_t bit=base+k;
        const uint8_t* sign0=B0+(bit>>3); const uint8_t* sign1=B1+(bit>>3);
        const __mmask16 mask0=(__mmask16)(unsigned(sign0[0])|(unsigned(sign0[1])<<8));
        const __mmask16 mask1=(__mmask16)(unsigned(sign1[0])|(unsigned(sign1[1])<<8));
        acc0=_mm512_add_ps(acc0,_mm512_mask_mov_ps(_mm512_sub_ps(_mm512_setzero_ps(),scaled0),mask0,scaled0));
        acc1=_mm512_add_ps(acc1,_mm512_mask_mov_ps(_mm512_sub_ps(_mm512_setzero_ps(),scaled1),mask1,scaled1));
      }
      O0[z]=_mm512_reduce_add_ps(acc0)*float(R0[n]);
      O1[z]=_mm512_reduce_add_ps(acc1)*float(R1[n]);
#elif defined(__ARM_NEON)
      const float acc0=wal_neon_binary_dot(X+m*K,B0,base,K,C0);
      const float acc1=wal_neon_binary_dot(X+m*K,B1,base,K,C1);
      O0[z]=acc0*float(R0[n]); O1[z]=acc1*float(R1[n]);
#else
      float acc0=0.0f,acc1=0.0f;
      for(int64_t k=0;k<K;++k){
        const float value=X[m*K+k];
        const int s0=int((B0[(base+k)>>3]>>((base+k)&7))&1)*2-1;
        const int s1=int((B1[(base+k)>>3]>>((base+k)&7))&1)*2-1;
        acc0+=value*float(C0[k])*float(s0); acc1+=value*float(C1[k])*float(s1);
      }
      O0[z]=acc0*float(R0[n]); O1[z]=acc1*float(R1[n]);
#endif
    }
  });
  return {out0,out1};
}

torch::Tensor t3_sparse_walb2_pair_cpu(
    torch::Tensor x, torch::Tensor t3, torch::Tensor pos,
    torch::Tensor sign_bits, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor hidden0, torch::Tensor hidden1,
    torch::Tensor u0, torch::Tensor u1, torch::Tensor row0, torch::Tensor row1) {
  x=x.contiguous().to(torch::kFloat32); t3=t3.contiguous(); pos=pos.contiguous();
  sign_bits=sign_bits.contiguous(); alpha=alpha.contiguous(); beta=beta.contiguous();
  hidden0=hidden0.contiguous(); hidden1=hidden1.contiguous();
  u0=u0.contiguous(); u1=u1.contiguous(); row0=row0.contiguous(); row1=row1.contiguous();
  const int64_t M=x.size(0),N=t3.size(0),K=t3.size(1)*4,G=K/128,BPR=K/4;
  const int64_t rank=hidden0.size(1);
  auto out=torch::empty({M,N},torch::TensorOptions().dtype(torch::kFloat32));
  const float* X=x.data_ptr<float>(); const uint8_t* T=t3.data_ptr<uint8_t>();
  const uint8_t* P=pos.data_ptr<uint8_t>(); const uint8_t* SB=sign_bits.data_ptr<uint8_t>();
  const at::Half* A=alpha.data_ptr<at::Half>(); const at::Half* E=beta.data_ptr<at::Half>();
  const float* H0=hidden0.data_ptr<float>(); const float* H1=hidden1.data_ptr<float>();
  const uint8_t* U0=u0.data_ptr<uint8_t>(); const uint8_t* U1=u1.data_ptr<uint8_t>();
  const at::Half* R0=row0.data_ptr<at::Half>(); const at::Half* R1=row1.data_ptr<at::Half>();
  float* O=out.data_ptr<float>();
  at::parallel_for(0,M*N,16,[&](int64_t begin,int64_t end){
    for(int64_t z=begin;z<end;++z){
      const int64_t m=z/N,n=z%N; float base_acc=0.0f;
      for(int64_t g=0;g<G;++g){
        const float* xv=X+m*K+g*128; const uint8_t* trow=T+n*BPR+g*32;
#if defined(__AVX512F__)
        __m512 densev=_mm512_setzero_ps();
        for(int k0=0;k0<128;k0+=16){
          unsigned positive=0,negative=0;
          for(int lane=0;lane<4;++lane){
            const uint8_t packed=trow[(k0>>2)+lane];
            positive|=unsigned(T3_MASKS.positive[packed])<<(lane*4);
            negative|=unsigned(T3_MASKS.negative[packed])<<(lane*4);
          }
          const __m512 value=_mm512_loadu_ps(xv+k0);
          densev=_mm512_mask_add_ps(densev,(__mmask16)positive,densev,value);
          densev=_mm512_mask_sub_ps(densev,(__mmask16)negative,densev,value);
        }
        const float dense=_mm512_reduce_add_ps(densev);
#elif defined(__ARM_NEON)
        const float dense=wal_neon_ternary_dot(xv,trow,128);
#else
        float dense=0.0f;
        for(int k=0;k<128;++k){
          const int code=int((trow[k>>2]>>((k&3)*2))&3)-1; dense+=xv[k]*float(code);
        }
#endif
        const int64_t q=(n*G+g)*8; const uint8_t signs=SB[n*G+g]; float sparse=0.0f;
        for(int j=0;j<8;++j) sparse+=xv[P[q+j]]*float(((signs>>j)&1)*2-1);
        base_acc+=dense*float(A[n*G+g])+sparse*float(E[n*G+g]);
      }
      const int64_t ubase=n*rank;
#if defined(__AVX512F__)
      __m512 wal0=_mm512_setzero_ps(),wal1=_mm512_setzero_ps();
      for(int64_t k=0;k<rank;k+=16){
        const __m512 h0=_mm512_loadu_ps(H0+m*rank+k);
        const __m512 h1=_mm512_loadu_ps(H1+m*rank+k);
        const uint8_t* s0=U0+((ubase+k)>>3); const uint8_t* s1=U1+((ubase+k)>>3);
        const __mmask16 mask0=(__mmask16)(unsigned(s0[0])|(unsigned(s0[1])<<8));
        const __mmask16 mask1=(__mmask16)(unsigned(s1[0])|(unsigned(s1[1])<<8));
        wal0=_mm512_add_ps(wal0,_mm512_mask_mov_ps(_mm512_sub_ps(_mm512_setzero_ps(),h0),mask0,h0));
        wal1=_mm512_add_ps(wal1,_mm512_mask_mov_ps(_mm512_sub_ps(_mm512_setzero_ps(),h1),mask1,h1));
      }
      const float correction0=_mm512_reduce_add_ps(wal0)*float(R0[n]);
      const float correction1=_mm512_reduce_add_ps(wal1)*float(R1[n]);
#elif defined(__ARM_NEON)
      float correction0=wal_neon_binary_dot(
          H0+m*rank,U0,ubase,rank,nullptr)*float(R0[n]);
      float correction1=wal_neon_binary_dot(
          H1+m*rank,U1,ubase,rank,nullptr)*float(R1[n]);
#else
      float correction0=0.0f,correction1=0.0f;
      for(int64_t k=0;k<rank;++k){
        const int s0=int((U0[(ubase+k)>>3]>>((ubase+k)&7))&1)*2-1;
        const int s1=int((U1[(ubase+k)>>3]>>((ubase+k)&7))&1)*2-1;
        correction0+=H0[m*rank+k]*float(s0); correction1+=H1[m*rank+k]*float(s1);
      }
      correction0*=float(R0[n]); correction1*=float(R1[n]);
#endif
      O[z]=base_acc+correction0+correction1;
    }
  }); return out;
}

torch::Tensor int4_packed_cpu(
    torch::Tensor x, torch::Tensor codes, torch::Tensor scales,
    int64_t N, int64_t K, int64_t group_size) {
  x=x.contiguous().to(torch::kFloat32); codes=codes.contiguous(); scales=scales.contiguous();
  const int64_t M=x.size(0),G=K/group_size,BPR=K/2;
  auto out=torch::empty({M,N},torch::TensorOptions().dtype(torch::kFloat32));
  const float* X=x.data_ptr<float>(); const uint8_t* C=codes.data_ptr<uint8_t>();
  const at::Half* S=scales.data_ptr<at::Half>(); float* O=out.data_ptr<float>();
  at::parallel_for(0,M*N,16,[&](int64_t begin,int64_t end){
    for(int64_t z=begin;z<end;++z){
      const int64_t m=z/N,n=z%N; float acc=0.0f;
      for(int64_t g=0;g<G;++g){ float dot=0.0f;
        const float* xv=X+m*K+g*group_size; const uint8_t* row=C+n*BPR+g*(group_size/2);
        #pragma omp simd reduction(+:dot)
        for(int64_t k=0;k<group_size;++k){
          const int code=int((row[k>>1]>>((k&1)*4))&15)-7;
          dot += xv[k]*float(code);
        }
        acc += dot*float(S[n*G+g]);
      } O[z]=acc;
    }
  }); return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("t3_sparse_packed_cpu",&t3_sparse_packed_cpu);
  m.def("binary_packed_cpu",&binary_packed_cpu);
  m.def("binary_pair_packed_cpu",&binary_pair_packed_cpu);
  m.def("t3_sparse_walb2_pair_cpu",&t3_sparse_walb2_pair_cpu);
  m.def("int4_packed_cpu",&int4_packed_cpu);
}
"""


_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        machine = platform.machine().lower().replace("-", "_")
        compiler_flags = ["-O3"]
        linker_flags = []
        if platform.system() == "Darwin":
            # Apple Clang has no system libgomp. ATen's native thread pool is
            # already used by at::parallel_for, so OpenMP is not required.
            compiler_flags.append("-mcpu=native")
        else:
            compiler_flags.extend(["-march=native", "-fopenmp"])
            linker_flags.append("-fopenmp")
        _EXTENSION = load_inline(
            f"wal_v77_direct_packed_cpu_v5_{machine}",
            cpp_sources=CPP,
            extra_cflags=compiler_flags,
            extra_ldflags=linker_flags,
            with_cuda=False,
            verbose=False,
        )
    return _EXTENSION


class DirectPackedCPUWALLinear(ref.PackedWALLinear):
    hardware_cache_root: Path | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._direct_cache = None

    def _direct(self):
        if self._direct_cache is None:
            if self.hardware_cache_root is None:
                raise ValueError("CPU direct runtime requires a hardware cache")
            base = load_hardware_base_cache(
                self.hardware_cache_root / (self.base_path.name + ".walhw"),
                torch.device("cpu"),
            )
            paths = () if self.overlay_path is None else load_packed_overlay(
                self.overlay_path, torch.device("cpu")
            )
            self._direct_cache = (base, paths)
        return self._direct_cache

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base, paths = self._direct()
        t3, positions, signs, alpha, beta = base[:5]
        flat = value.reshape(-1, value.shape[-1])
        if len(paths) == 2:
            path0, path1 = paths
            hidden0, hidden1 = extension().binary_pair_packed_cpu(
                flat, path0["v"], path1["v"], path0["rank"], path0["in"],
                path0["latent"], path1["latent"],
                path0["column"], path1["column"],
            )
            result = extension().t3_sparse_walb2_pair_cpu(
                flat, t3, positions, signs, alpha, beta,
                hidden0, hidden1, path0["u"], path1["u"],
                path0["row"], path1["row"],
            )
            return result.reshape(
                *value.shape[:-1], self.out_features,
            ).to(self.output_dtype)
        result = extension().t3_sparse_packed_cpu(
            flat, t3, positions, signs, alpha, beta
        )
        for path in paths:
            hidden = extension().binary_packed_cpu(
                flat, path["v"], path["rank"], path["in"],
                path["latent"], path["column"],
            )
            result.add_(extension().binary_packed_cpu(
                hidden, path["u"], path["out"], path["rank"],
                path["row"], torch.empty(0, dtype=torch.float16),
            ))
        return result.reshape(*value.shape[:-1], self.out_features).to(self.output_dtype)


class DirectPackedCPUINT4Head(torch.nn.Module):
    def __init__(self, path: Path, *, endpoint, row_chunk: int, output_dtype):
        super().__init__()
        self.path = Path(path)
        self.endpoint = endpoint
        header = endpoint._read_endpoint_header(self.path)
        self.spec = header["spec"]
        self.payload = header["payload"]
        if self.spec.name != "int4-symmetric" or int(self.spec.code_bits) != 4:
            raise ValueError("DirectPackedCPUINT4Head requires int4-symmetric")
        self.in_features = int(self.payload.columns)
        self.out_features = int(self.payload.rows)
        self.output_dtype = output_dtype
        self._packed_cache = {}

    def _packed(self, device: torch.device):
        key = str(device)
        cached = self._packed_cache.get(key)
        if cached is None:
            header_bytes = self.endpoint.ENDPOINT_HEADER.size
            code_bytes = int(self.payload.code_bits) // 8
            metadata_bytes = int(self.payload.metadata_bits) // 8
            with self.path.open("rb") as handle:
                handle.seek(header_bytes)
                raw_codes = handle.read(code_bytes)
                raw_scales = handle.read(metadata_bytes)
                if len(raw_codes) != code_bytes or len(raw_scales) != metadata_bytes:
                    raise ValueError("truncated direct INT4 endpoint")
                if handle.read(1):
                    raise ValueError("direct INT4 endpoint has trailing bytes")
            codes = torch.frombuffer(bytearray(raw_codes), dtype=torch.uint8).to(device)
            scales = torch.frombuffer(bytearray(raw_scales), dtype=torch.float16).to(device)
            if not torch.isfinite(scales).all() or torch.any(scales <= 0):
                raise ValueError("invalid direct INT4 endpoint scales")
            cached = (codes.contiguous(), scales.contiguous())
            self._packed_cache[key] = cached
        return cached

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        codes, scales = self._packed(torch.device("cpu"))
        flat = value.reshape(-1, value.shape[-1])
        result = extension().int4_packed_cpu(
            flat, codes, scales, self.out_features, self.in_features,
            int(self.payload.group_size),
        )
        return result.reshape(*value.shape[:-1], self.out_features).to(self.output_dtype)


def rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux/BSD.
    return int(usage if platform.system() == "Darwin" else usage * 1024)


def load_model(checkpoint: Path, hardware_cache: Path):
    extension()
    DirectPackedCPUWALLinear.hardware_cache_root = hardware_cache.resolve(strict=True)
    return ref.load_packed_model(
        checkpoint, device="cpu", dtype=torch.float32, row_chunk=256,
        verify_hashes=False, wal_linear_class=DirectPackedCPUWALLinear,
        endpoint_linear_class=DirectPackedCPUINT4Head,
    )


@torch.no_grad()
def main() -> None:
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=str(ref.DEFAULT_CHECKPOINT))
    parser.add_argument("--hardware-cache", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started=time.monotonic(); rss_before=rss_bytes()
    model,report=load_model(Path(args.checkpoint).resolve(strict=True),Path(args.hardware_cache))
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    inputs=tokenizer(args.prompt,return_tensors="pt")
    cold=time.monotonic(); logits=model(**inputs,use_cache=False).logits[:,-1].float(); cold_s=time.monotonic()-cold
    warm=time.monotonic(); logits2=model(**inputs,use_cache=False).logits[:,-1].float(); warm_s=time.monotonic()-warm
    report.update({
        "runtime_tier":"direct-packed-native-cpu-v4-fused-walb2",
        "threads":args.threads,
        "cold_forward_seconds":cold_s,
        "warm_forward_seconds":warm_s,
        "warm_tokens_per_second":inputs.input_ids.numel()/warm_s,
        "argmax_token_id":int(logits.argmax(-1).item()),
        "argmax_token":tokenizer.decode(logits.argmax(-1).tolist()),
        "warm_argmax_matches":bool(logits.argmax(-1).equal(logits2.argmax(-1))),
        "rss_before_bytes":rss_before,
        "rss_after_bytes":rss_bytes(),
        "persistent_int8_or_fp_weight_cache":False,
        "total_seconds":time.monotonic()-started,
    })
    raw=(json.dumps(report,indent=2,allow_nan=False)+"\n").encode()
    descriptor=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o444)
    try:
        offset=0
        while offset<len(raw): offset+=os.write(descriptor,raw[offset:])
        os.fsync(descriptor)
    finally: os.close(descriptor)
    print(json.dumps(report,indent=2,allow_nan=False))
    del model; gc.collect()


if __name__ == "__main__":
    main()
