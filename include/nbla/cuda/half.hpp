// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NBLA_CUDA_HALF_HPP_
#define NBLA_CUDA_HALF_HPP_
#include <cmath>
#include <nbla/half.hpp>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
#if CUDA_VERSION >= 7050
#define NBLA_CUDA_HALF 1
#else
#error "NNabla doesn't support CUDA version less than 7.5."
#endif
#else
#define NBLA_CUDA_HALF 0
#endif
#if NBLA_CUDA_HALF
#define HALF_CUDA_PREFIX __device__ __forceinline__
#define HALF_CUDA_HOSTDEVICE_PREFIX __device__ __host__ __forceinline__
#else // !NBLA_CUDA_HALF
#define HALF_CUDA_PREFIX inline
#define HALF_CUDA_HOSTDEVICE_PREFIX inline
#endif // NBLA_CUDA_HALF

namespace nbla {

/** Operator overloaded class for CUDA half type.
 */
struct NBLA_ALIGN(2) HalfCuda {
  half h;
  // ----------------------------------------------------------------------------
  // Constructors
  // ----------------------------------------------------------------------------
  HALF_CUDA_HOSTDEVICE_PREFIX HalfCuda() {}
#if CUDA_VERSION >= 9000
  HALF_CUDA_HOSTDEVICE_PREFIX HalfCuda(const __half_raw &rhs) { h = rhs; }
#endif
  HALF_CUDA_HOSTDEVICE_PREFIX HalfCuda(const Half &rhs) {
#if CUDA_VERSION >= 9000
    h = __half_raw{rhs.bits};
#else
    h.x = rhs.bits;
#endif
  }
  HALF_CUDA_HOSTDEVICE_PREFIX HalfCuda(const HalfCuda &rhs) { h = rhs.h; }
  HALF_CUDA_HOSTDEVICE_PREFIX operator Half() const {
    Half cpu_h;
#if CUDA_VERSION >= 9000
    cpu_h.bits = ((__half_raw)h).x;
#else
    cpu_h.bits = h.x;
#endif
    return cpu_h;
  }
  HALF_CUDA_PREFIX HalfCuda(const half &rhs) : h(rhs) {}
  HALF_CUDA_PREFIX HalfCuda &operator=(const HalfCuda &rhs) {
    h = rhs.h;
    return *this;
  }
#if NBLA_CUDA_HALF
  HALF_CUDA_PREFIX unsigned short as_bits() const {
#if CUDA_VERSION >= 9000
    return ((__half_raw)h).x;
#else
    return h.x;
#endif
  }
#if CUDA_VERSION >= 9000
#define STORE_FLOAT2HALF_RN(F) h = __float2half_rn(F)
#else // CUDA_VERSION >= 9000
#define STORE_FLOAT2HALF_RN(F) h.x = __float2half_rn(F)
#endif
  HALF_CUDA_PREFIX HalfCuda(float f) { STORE_FLOAT2HALF_RN(f); }

// ----------------------------------------------------------------------------
// Cast
// ----------------------------------------------------------------------------
#if CUDA_VERSION >= 9000
  HALF_CUDA_PREFIX operator float() const { return (float)(h); }
  HALF_CUDA_PREFIX operator short() const { return (short)(h); }
  HALF_CUDA_PREFIX operator int() const { return (int)(h); }
  HALF_CUDA_PREFIX operator long() const { return (long)((int)h); }
  HALF_CUDA_PREFIX operator unsigned short() const {
    return (unsigned short)(h);
  }
  HALF_CUDA_PREFIX operator unsigned int() const { return (unsigned int)(h); }
  HALF_CUDA_PREFIX operator unsigned long() const {
    return (unsigned long)((unsigned int)h);
  }
  HALF_CUDA_PREFIX operator unsigned long long() const {
    return (unsigned long long)((unsigned short)h);
  }
  HALF_CUDA_PREFIX operator bool() const { return (bool)(h); }
  HALF_CUDA_PREFIX operator double() const { return (double)(float(*this)); }
  HALF_CUDA_PREFIX operator char() const { return (char)((short)(*this)); }
  HALF_CUDA_PREFIX operator unsigned char() const {
    return (unsigned char)((unsigned short)(*this));
  }
#else // CUDA_VERSION < 9000
  HALF_CUDA_PREFIX operator float() const { return __half2float(h); }
  HALF_CUDA_PREFIX operator short() const { return (short)((float)(*this)); }
  HALF_CUDA_PREFIX operator int() const { return (int)((float)(*this)); }
  HALF_CUDA_PREFIX operator long() const { return (long)((float)(*this)); }
  HALF_CUDA_PREFIX operator unsigned short() const {
    return (unsigned short)((float)(*this));
  }
  HALF_CUDA_PREFIX operator unsigned int() const {
    return (unsigned int)((float)(*this));
  }
  HALF_CUDA_PREFIX operator unsigned long() const {
    return (unsigned long)((float)(*this));
  }
  HALF_CUDA_PREFIX operator unsigned long long() const {
    return (unsigned long long)((float)(*this));
  }
  HALF_CUDA_PREFIX operator bool() const { return (bool)((float)(*this)); }
  HALF_CUDA_PREFIX operator double() const { return (double)(float(*this)); }
  HALF_CUDA_PREFIX operator char() const { return (char)((float)(*this)); }
  HALF_CUDA_PREFIX operator unsigned char() const {
    return (unsigned char)((float)(*this));
  }
#endif
  // ----------------------------------------------------------------------------

  // ----------------------------------------------------------------------------

  // ----------------------------------------------------------------------------
  // Arithmetic operators
  // ----------------------------------------------------------------------------
  HALF_CUDA_PREFIX HalfCuda operator+() const { return *this; }
  HALF_CUDA_PREFIX HalfCuda operator-() const {
#if __CUDA_ARCH__ >= 530
    return HalfCuda{__hneg(h)};
#else
    return HalfCuda{-(float)(*this)};
#endif
  }
#define AOP_H(OP, INST)                                                        \
  return HalfCuda { INST(h, rhs.h) }
#define AOP_F(OP, INST)                                                        \
  return HalfCuda { __half2float(h) OP __half2float(rhs.h) }
#if __CUDA_ARCH__ >= 530
#define AOP_ AOP_H
#else
#define AOP_ AOP_F
#endif
#define AOP(OP, INST)                                                          \
  HALF_CUDA_PREFIX HalfCuda operator OP(const HalfCuda &rhs) const {           \
    AOP_(OP, INST);                                                            \
  }
  AOP(+, __hadd);
  AOP(-, __hsub);
  AOP(*, __hmul);
  HALF_CUDA_PREFIX HalfCuda operator/(const HalfCuda &rhs) const {
    AOP_F(/, __hdiv);
  }
#undef AOP
#undef AOP_
#define AOP(OP, TYPE)                                                          \
  HALF_CUDA_PREFIX TYPE operator OP(const TYPE &rhs) const {                   \
    return __half2float(h) OP rhs;                                             \
  }

#define AOPS(TYPE)                                                             \
  AOP(+, TYPE);                                                                \
  AOP(-, TYPE);                                                                \
  AOP(*, TYPE);                                                                \
  AOP(/, TYPE)
  AOPS(float);
  AOPS(double);
#undef AOP
#undef AOPS
#define AOP(OP, TYPE)                                                          \
  HALF_CUDA_PREFIX HalfCuda operator OP(const TYPE &rhs) const {               \
    return *this OP HalfCuda(rhs);                                             \
  }

#define AOPS(TYPE)                                                             \
  AOP(+, TYPE);                                                                \
  AOP(-, TYPE);                                                                \
  AOP(*, TYPE);                                                                \
  AOP(/, TYPE)
  AOPS(bool);
  AOPS(unsigned char);
  AOPS(unsigned short);
  AOPS(unsigned int);
  AOPS(char);
  AOPS(short);
  AOPS(int);
#undef AOP
#undef AOPS
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// In-place arithmetic operators
// ----------------------------------------------------------------------------
#define IAOP_H(OP, INST)                                                       \
  HALF_CUDA_PREFIX HalfCuda &operator OP(const HalfCuda &rhs) {                \
    h = INST(h, rhs.h);                                                        \
    return *this;                                                              \
  }
#define IAOP_F(OP, INST)                                                       \
  HALF_CUDA_PREFIX HalfCuda &operator OP(const HalfCuda &rhs) {                \
    float tmp = (float)(*this);                                                \
    tmp OP(float)(rhs);                                                        \
    STORE_FLOAT2HALF_RN(tmp);                                                  \
    return *this;                                                              \
  }
#if __CUDA_ARCH__ >= 530
#define IAOP IAOP_H
#if CUDA_VERSION >= 9000
#define IAOP_DIV IAOP_H
#else // CUDA_VERSION >= 9000
#define IAOP_DIV IAOP_F
#endif // CUDA_VERSION >= 9000
#else  // !(__CUDA_ARCH__ >= 530)
#define IAOP IAOP_F
#define IAOP_DIV IAOP_F
#endif // __CUDA_ARCH__ >= 530
  IAOP(+=, __hadd);
  IAOP(-=, __hsub);
  IAOP(*=, __hmul);
  IAOP_DIV(/=, __hdiv);

// ----------------------------------------------------------------------------
#undef IAOP
#undef STORE_FLOAT2HALF_RN
#endif // NBLA_CUDA_HALF
};

#if NBLA_CUDA_HALF
// ----------------------------------------------------------------------------
// Inverse arithmetic operators
// ----------------------------------------------------------------------------
#define AOP(OP, TYPE)                                                          \
  HALF_CUDA_PREFIX HalfCuda operator OP(const TYPE &lhs,                       \
                                        const HalfCuda &rhs) {                 \
    return HalfCuda(lhs) OP rhs;                                               \
  }
#define AOPS(TYPE)                                                             \
  AOP(+, TYPE);                                                                \
  AOP(-, TYPE);                                                                \
  AOP(*, TYPE);                                                                \
  AOP(/, TYPE)
AOPS(bool);
AOPS(unsigned char);
AOPS(unsigned short);
AOPS(unsigned int);
AOPS(char);
AOPS(short);
AOPS(int);
#undef AOP
#undef AOPS
#define AOP(OP, TYPE)                                                          \
  HALF_CUDA_PREFIX TYPE operator OP(const TYPE &lhs, const HalfCuda &rhs) {    \
    return lhs OP(float) rhs;                                                  \
  }
#define AOPS(TYPE)                                                             \
  AOP(+, TYPE);                                                                \
  AOP(-, TYPE);                                                                \
  AOP(*, TYPE);                                                                \
  AOP(/, TYPE);
AOPS(float);
AOPS(double);
#undef AOP
#undef AOPS
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Relational operators
// ----------------------------------------------------------------------------
// Relational operators
#define ROP_TYPE(OP, TYPE)                                                     \
  HALF_CUDA_PREFIX bool operator OP(const HalfCuda &lhs, const TYPE &rhs)
#define IROP_TYPE(OP, TYPE)                                                    \
  HALF_CUDA_PREFIX bool operator OP(const TYPE &lhs, const HalfCuda &rhs)
#define ROP(TYPE)                                                              \
  ROP_TYPE(<, TYPE) { return (float)lhs < rhs; }                               \
  IROP_TYPE(<, TYPE) { return lhs < (float)rhs; }                              \
  ROP_TYPE(>, TYPE) { return rhs < lhs; }                                      \
  IROP_TYPE(>, TYPE) { return rhs < lhs; }                                     \
  ROP_TYPE(<=, TYPE) { return !(lhs > rhs); }                                  \
  ROP_TYPE(>=, TYPE) { return !(lhs < rhs); }                                  \
  ROP_TYPE(==, TYPE) { return (float)lhs == rhs; }                             \
  ROP_TYPE(!=, TYPE) { return !(lhs == rhs); }                                 \
  IROP_TYPE(<=, TYPE) { return !(lhs > rhs); }                                 \
  IROP_TYPE(>=, TYPE) { return !(lhs < rhs); }
ROP(unsigned char);
ROP(char);
ROP(unsigned short);
ROP(short);
ROP(unsigned int);
ROP(int);
// ROP(unsigned long);
// ROP(long);
// ROP(unsigned long long);
// ROP(long long);
ROP(float);
ROP(double);
ROP(bool);
// ROP(long double);
#if __CUDA_ARCH__ >= 530
ROP_TYPE(<, HalfCuda) { return __hlt(lhs.h, rhs.h); }
ROP_TYPE(>, HalfCuda) { return rhs < lhs; }
ROP_TYPE(<=, HalfCuda) { return !(lhs > rhs); }
ROP_TYPE(>=, HalfCuda) { return !(lhs < rhs); }
ROP_TYPE(==, HalfCuda) { return __heq(lhs.h, rhs.h); }
ROP_TYPE(!=, HalfCuda) { return !(lhs == rhs); }
#else
ROP_TYPE(<, HalfCuda) { return (float)lhs < (float)rhs; }
ROP_TYPE(>, HalfCuda) { return rhs < lhs; }
ROP_TYPE(<=, HalfCuda) { return !(lhs > rhs); }
ROP_TYPE(>=, HalfCuda) { return !(lhs < rhs); }
ROP_TYPE(==, HalfCuda) { return (float)lhs == (float)rhs; }
ROP_TYPE(!=, HalfCuda) { return !(lhs == rhs); }
#endif
#undef ROP_TYPE
#undef IROP_TYPE
#undef ROP
#endif // NBLA_CUDA_HALF

// ----------------------------------------------------------------------------

/** Infer NNabla's CUDA type.

    When nbla::Half is passed, it's converted to nbla::HalfCuda which can be
   used in kernel functions as if it's a built-in scalar type with overloaded
   operators.
 */
template <typename T> struct CudaType { typedef T type; };
template <> struct CudaType<Half> { typedef HalfCuda type; };
/** Infer NNabla's CUDA type while force half to float.

    This is used when a particular operation doesn't support fp16 computation
   (e.g. GEMV in cuBLAS at least until ver 9.1)
 */
template <typename T> struct CudaTypeForceFloat { typedef T type; };
template <> struct CudaTypeForceFloat<Half> { typedef float type; };
template <> struct CudaTypeForceFloat<HalfCuda> { typedef float type; };
template <> struct CudaTypeForceFloat<half> { typedef float type; };

/** Infer CUDA's native data type from NNabla's data type.

    In particular, nbla::Half and nbla::HalfCuda are converted to CUDA's half.
   Otherwise passed through.
*/
template <typename T> struct CudaNativeType { typedef T type; };
template <> struct CudaNativeType<Half> { typedef half type; };
template <> struct CudaNativeType<HalfCuda> { typedef half type; };

/** Get a scalar value of CUDA's native type from NNabla's data type.
*/
template <typename T>
typename CudaNativeType<T>::type get_native_arg(const T &v) {
  return v;
}
template <>
inline typename CudaNativeType<HalfCuda>::type
get_native_arg<HalfCuda>(const HalfCuda &v) {
  return v.h;
}

/** Template specialization for nbla::HalfCuda of a function nbla::get_dtype
    which gets an enum value of nbla::dtypes.
*/
template <> inline dtypes get_dtype<HalfCuda>() { return dtypes::HALF; }

/** Return CUDA's scalar value from float.

    Returns as a type  corresponding to the specified template argument.
   Specifically, Half --> half, HalfCuda --> half.
*/
template <class T>
typename CudaNativeType<T>::type get_cuda_native_scalar(float val) {
  return val;
}
template <>
inline typename CudaNativeType<Half>::type
get_cuda_native_scalar<Half>(float val) {
  return HalfCuda(Half(val)).h;
}
template <>
inline typename CudaNativeType<HalfCuda>::type
get_cuda_native_scalar<HalfCuda>(float val) {
  return HalfCuda(Half(val)).h;
}

} // End of nbla

#if NBLA_CUDA_HALF
// ----------------------------------------------------------------------------
// cmath functions
// ----------------------------------------------------------------------------
namespace std {
using namespace nbla;
#define MATHF_F(FUNC)                                                          \
  HALF_CUDA_PREFIX HalfCuda FUNC(const HalfCuda &h) {                          \
    return HalfCuda{std::FUNC((float)h)};                                      \
  }
#if __CUDA_ARCH__ >= 530
#define MATHF(FUNC)                                                            \
  HALF_CUDA_PREFIX HalfCuda FUNC(const HalfCuda &h) {                          \
    return HalfCuda{h##FUNC(h.h)};                                             \
  }
#else
#define MATHF(FUNC) MATHF_F(FUNC)
#endif
MATHF_F(abs);
MATHF_F(fabs);
MATHF_F(tanh);
MATHF(exp);
MATHF(log);
// MATHF_F(log2);
MATHF(sqrt);
MATHF(floor);
MATHF(ceil);
HALF_CUDA_PREFIX HalfCuda pow(const HalfCuda &a, const HalfCuda &b) {
  return std::pow((float)a, (float)b);
}
HALF_CUDA_PREFIX HalfCuda pow(const HalfCuda &a, int &b) {
  return std::pow((float)a, b);
}
HALF_CUDA_PREFIX nbla::HalfCuda atan2(const nbla::HalfCuda &a,
                                      const nbla::HalfCuda &b) {
  return std::atan2((float)a, (float)b);
}
#undef MATHF
#undef MATHF_F
} // End of std
#define MATHF_F(FUNC)                                                          \
  HALF_CUDA_PREFIX nbla::HalfCuda FUNC(const nbla::HalfCuda &h) {              \
    return nbla::HalfCuda{FUNC((float)h)};                                     \
  }
#if __CUDA_ARCH__ >= 530
#define MATHF(FUNC)                                                            \
  HALF_CUDA_PREFIX nbla::HalfCuda FUNC(const nbla::HalfCuda &h) {              \
    return nbla::HalfCuda{h##FUNC(h.h)};                                       \
  }
#else
#define MATHF(FUNC) MATHF_F(FUNC)
#endif
MATHF_F(abs);
MATHF_F(fabs);
MATHF_F(sin);
MATHF_F(cos);
MATHF_F(tan);
MATHF_F(sinh);
MATHF_F(cosh);
MATHF_F(tanh);
MATHF_F(asin);
MATHF_F(acos);
MATHF_F(atan);
MATHF_F(asinh);
MATHF_F(acosh);
MATHF_F(atanh);
MATHF_F(round);
MATHF(exp);
MATHF(log);
MATHF(sqrt);
MATHF(rsqrt);
MATHF(floor);
MATHF(ceil);
HALF_CUDA_PREFIX nbla::HalfCuda max(const nbla::HalfCuda &a,
                                    const nbla::HalfCuda &b) {
  return a > b ? a : b;
}

HALF_CUDA_PREFIX nbla::HalfCuda min(const nbla::HalfCuda &a,
                                    const nbla::HalfCuda &b) {
  return a < b ? a : b;
}
HALF_CUDA_PREFIX nbla::HalfCuda pow(const nbla::HalfCuda &a,
                                    const nbla::HalfCuda &b) {
  return pow((float)a, (float)b);
}
HALF_CUDA_PREFIX nbla::HalfCuda pow(const nbla::HalfCuda &a, int &b) {
  return pow((float)a, b);
}
HALF_CUDA_PREFIX int isnan(const nbla::HalfCuda &x) {
  return (x.as_bits() & 0x7FFF) > 0x7C00;
}
HALF_CUDA_PREFIX int isinf(const nbla::HalfCuda &x) {
  return (x.as_bits() & 0x7FFF) == 0x7C00;
}
HALF_CUDA_PREFIX nbla::HalfCuda atan2(const nbla::HalfCuda &a,
                                      const nbla::HalfCuda &b) {
  return atan2((float)a, (float)b);
}
#endif // NBLA_CUDA_HALF

// ----------------------------------------------------------------------------
#endif
