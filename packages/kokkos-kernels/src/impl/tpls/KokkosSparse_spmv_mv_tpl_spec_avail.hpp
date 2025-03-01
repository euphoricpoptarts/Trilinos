/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSPARSE_SPMV_MV_TPL_SPEC_AVAIL_HPP_
#define KOKKOSPARSE_SPMV_MV_TPL_SPEC_AVAIL_HPP_

namespace KokkosSparse {
namespace Impl {

// Specialization struct which defines whether a specialization exists
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM,
          const bool integerScalarType =
              std::is_integral<typename std::decay<AT>::type>::value>
struct spmv_mv_tpl_spec_avail {
  enum : bool { value = false };
};

#define KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(SCALAR, ORDINAL, OFFSET, \
                                                     XL, YL, MEMSPACE)        \
  template <>                                                                 \
  struct spmv_mv_tpl_spec_avail<                                              \
      const SCALAR, const ORDINAL, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET, const SCALAR**,  \
      XL, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                             \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,         \
      SCALAR**, YL, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                   \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > {                             \
    enum : bool { value = true };                                             \
  };

/* CUSPARSE_VERSION 10300 and lower seem to have a bug in cusparseSpMM
non-transpose that produces incorrect result. This is cusparse distributed with
CUDA 10.1.243. The bug seems to be resolved by CUSPARSE 10301 (present by
CUDA 10.2.89) */
#if defined(CUSPARSE_VERSION) && (10301 <= CUSPARSE_VERSION)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(double, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(float, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<double>, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int,
                                             Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::complex<float>, int, int,
                                             Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int,
                                             int, Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int,
                                             int, Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaSpace)

KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int,
                                             int, Kokkos::LayoutLeft,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_MV_TPL_SPEC_AVAIL_CUSPARSE(Kokkos::Experimental::half_t, int,
                                             int, Kokkos::LayoutRight,
                                             Kokkos::LayoutLeft,
                                             Kokkos::CudaUVMSpace)

#endif
#endif  // defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSPARSE_SPMV_MV_TPL_SPEC_AVAIL_HPP_
