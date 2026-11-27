#pragma once

#include <array>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

template<typename T> using opmath_t = at::opmath_type<T>;

constexpr int64_t kChunkSize = 16;

// Vector step of CPU kernels whose NEON results must match AVX2, in registers
// and in scalars of T: a register pair on NEON, whose registers are half as wide
// as AVX2's, and a single register elsewhere.
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
constexpr int kParityVecRegs = 2;
#else
constexpr int kParityVecRegs = 1;
#endif

template <typename T>
constexpr int64_t kParityVecSize = vec::Vectorized<T>::size() * kParityVecRegs;

// Each Welford accumulator spans one parity step.
template <typename T>
using MomentsVec = std::conditional_t<
    kParityVecRegs == 1,
    vec::Vectorized<opmath_t<T>>,
    vec::VectorizedN<opmath_t<T>, kParityVecRegs>>;

template <typename T>
void AddMoments(
    int64_t m0_add,
    const T& m1_add,
    const T& m2_add,
    int64_t& m0,
    T& m1,
    T& m2) {
  const int64_t n = m0 + m0_add;
  const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  const T delta = m1_add - m1;
  m1 += c * delta;
  m2 += m2_add + delta * delta * c * static_cast<T>(m0);
  m0 = n;
}

template <typename Vec>
C10_ALWAYS_INLINE void AddMomentsVec(
    int64_t m0_add,
    const Vec& m1_add,
    const Vec& m2_add,
    int64_t& m0,
    Vec& m1,
    Vec& m2) {
  using T = typename Vec::value_type;
  const int64_t n = m0 + m0_add;
  const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  const Vec c_vec(c);
  const Vec delta = m1_add - m1;
  const Vec m2_tmp = m2 + m2_add;
  const Vec c_vec_delta = c_vec * delta;
  const Vec m0_delta = delta * Vec(static_cast<T>(m0));
  m1 = m1 + c_vec_delta;
  m2 = fmadd(m0_delta, c_vec_delta, m2_tmp);
  m0 = n;
}

template <typename T, typename acc_t = opmath_t<T>>
inline std::enable_if_t<std::is_same_v<T, acc_t>, void>
UpdateMomentsVec(
    int64_t m0,
    const T* X_ptr,
    const std::array<MomentsVec<acc_t>, kChunkSize>& c_vecs,
    int64_t& m0_stk0,
    MomentsVec<acc_t>& m1_stk0,
    MomentsVec<acc_t>& m2_stk0) {
  using Vec = MomentsVec<acc_t>;
  Vec m1_vec(0);
  Vec m2_vec(0);
  for (const auto j : c10::irange(m0)) {
    const Vec x_vec = Vec::loadu(X_ptr + j * Vec::size());
    const Vec tmpVec = c_vecs[j];
    const Vec delta_vec = x_vec - m1_vec;
    m1_vec = fmadd(tmpVec, delta_vec, m1_vec);
    const Vec tmpVec2 = x_vec - m1_vec;
    m2_vec = fmadd(delta_vec, tmpVec2, m2_vec);
  }
  AddMomentsVec(m0, m1_vec, m2_vec, m0_stk0, m1_stk0, m2_stk0);
}

// Loads kParityVecSize<T> scalars widened to acc_t as two MomentsVec<acc_t>:
// the first holds scalars [0, kParityVecSize<T> / 2), the second the rest.
template <typename T, typename acc_t = opmath_t<T>>
inline std::tuple<MomentsVec<acc_t>, MomentsVec<acc_t>> LoadMomentsVec(const T* ptr) {
  using Vec = vec::Vectorized<T>;
  if constexpr (std::is_same_v<T, float> && std::is_same_v<acc_t, double>) {
    using Vec1 = vec::VectorizedN<float, 1>;
    if constexpr (kParityVecRegs == 1) {
      const auto wide{vec::convert<double, 2, float, 1>(Vec1{Vec::loadu(ptr)})};
      return {wide[0], wide[1]};
    } else {
      return {vec::convert<double, 2, float, 1>(Vec1{Vec::loadu(ptr)}),
              vec::convert<double, 2, float, 1>(Vec1{Vec::loadu(ptr + Vec::size())})};
    }
  } else if constexpr (kParityVecRegs == 1) {
    return convert_to_float<T>(Vec::loadu(ptr));
  } else {
    auto [x0, x1] = convert_to_float<T>(Vec::loadu(ptr));
    auto [x2, x3] = convert_to_float<T>(Vec::loadu(ptr + Vec::size()));
    return {MomentsVec<T>(x0, x1), MomentsVec<T>(x2, x3)};
  }
}

// Vectors of T will be widened into two acc_t vectors,
// and accumulated successively on m1_stk0/m2_stk0.
template <typename T, typename acc_t = opmath_t<T>>
inline std::enable_if_t<!std::is_same_v<T, acc_t>, void>
UpdateMomentsVec(
    int64_t m0,
    const T* X_ptr,
    const std::array<MomentsVec<acc_t>, kChunkSize>& c_vecs,
    int64_t& m0_stk0,
    MomentsVec<acc_t>& m1_stk0,
    MomentsVec<acc_t>& m2_stk0) {
  using fVec = MomentsVec<acc_t>;
  static_assert(kParityVecSize<T> == 2 * fVec::size());
  fVec m1_fvec0(0), m1_fvec1(0);
  fVec m2_fvec0(0), m2_fvec1(0);
  for (const auto j : c10::irange(m0)) {
    const fVec tmpVec = c_vecs[j];
    auto [x_fvec0, x_fvec1] = LoadMomentsVec<T, acc_t>(X_ptr + j * kParityVecSize<T>);
    const fVec delta_fvec0 = x_fvec0 - m1_fvec0;
    const fVec delta_fvec1 = x_fvec1 - m1_fvec1;
    m1_fvec0 = fmadd(delta_fvec0, tmpVec, m1_fvec0);
    m1_fvec1 = fmadd(delta_fvec1, tmpVec, m1_fvec1);
    const fVec delta_fvec2 = x_fvec0 - m1_fvec0;
    const fVec delta_fvec3 = x_fvec1 - m1_fvec1;
    m2_fvec0 = fmadd(delta_fvec0, delta_fvec2, m2_fvec0);
    m2_fvec1 = fmadd(delta_fvec1, delta_fvec3, m2_fvec1);
  }
  AddMomentsVec(m0, m1_fvec0, m2_fvec0, m0_stk0, m1_stk0, m2_stk0);
  AddMomentsVec(m0, m1_fvec1, m2_fvec1, m0_stk0, m1_stk0, m2_stk0);
}

// Compute rowwise moments by Welford algorithm and cascade sum to improve
// numerical stability.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
// https://en.wikipedia.org/wiki/Pairwise_summation
template <typename T, int64_t kMaxDepth, typename acc_t>
std::pair<acc_t, acc_t> RowwiseMomentsImpl(const T* X, int64_t N) {
  using Vec = MomentsVec<acc_t>;
  constexpr int64_t kVecSize = kParityVecSize<T>;
  constexpr int64_t kAccVecSize = Vec::size();
  const int64_t n = N / kVecSize;
  const int64_t m = divup(n, kChunkSize);
  const int64_t depth = utils::CeilLog2(m);

  const Vec kZeroVec(acc_t(0));
  std::array<int64_t, kMaxDepth> m0_stk = {{0}};
  std::array<Vec, kMaxDepth> m1_stk;
  m1_stk.fill(kZeroVec);
  std::array<Vec, kMaxDepth> m2_stk;
  m2_stk.fill(kZeroVec);

  for (const auto i : c10::irange(m)) {
    const T* X_ptr = X + i * kChunkSize * kVecSize;
    const int64_t m0 = std::min(kChunkSize, n - i * kChunkSize);
    static std::array<Vec, kChunkSize> c_vecs = ([]() {
      std::array<Vec, kChunkSize> result;
      for (const auto i : c10::irange(kChunkSize)) {
        result[i] = Vec(acc_t(1) / static_cast<acc_t>(i + 1));
      }
      return result;
    })();
    UpdateMomentsVec<T, acc_t>(m0, X_ptr, c_vecs, m0_stk[0], m1_stk[0], m2_stk[0]);

    int64_t mask = i + 1;
    for (int64_t j = 1; j < depth && (mask & 1) == 0; ++j) {
      AddMomentsVec(
          m0_stk[j - 1],
          m1_stk[j - 1],
          m2_stk[j - 1],
          m0_stk[j],
          m1_stk[j],
          m2_stk[j]);
      m0_stk[j - 1] = 0;
      m1_stk[j - 1] = kZeroVec;
      m2_stk[j - 1] = kZeroVec;
      mask >>= 1;
    }
  }
  for (const auto i : c10::irange(1, depth)) {
    AddMomentsVec(
        m0_stk[i], m1_stk[i], m2_stk[i], m0_stk[0], m1_stk[0], m2_stk[0]);
  }

  std::array<acc_t, kAccVecSize> m1_arr{};
  std::array<acc_t, kAccVecSize> m2_arr{};
  m1_stk[0].store(m1_arr.data());
  m2_stk[0].store(m2_arr.data());

  int64_t m0 = 0;
  acc_t m1 = 0;
  acc_t m2 = 0;
  for (int64_t i = n * kVecSize; i < N; ++i) {
    acc_t x = static_cast<acc_t>(X[i]);
    const acc_t delta = x - m1;
    ++m0;
    m1 += delta / static_cast<acc_t>(m0);
    m2 += delta * (x - m1);
  }
  // Without a full vector step every lane is empty, and merging one computes
  // 0 * delta, which is NaN for an infinite tail.
  if (n > 0) {
    // When T is widened to acc_t, each lane in m1_arr/m2_arr holds 2*n accumulated results
    const int64_t m0_add{n * kVecSize / kAccVecSize};
    for (const auto i : c10::irange(kAccVecSize)) {
      AddMoments(m0_add, m1_arr[i], m2_arr[i], m0, m1, m2);
    }
  }

  return std::make_pair(m1, m2);
}

// Mean and M2 (sum of squared deviations from the mean) of X[0, N), accumulated in acc_t.
template <typename T, typename acc_t = opmath_t<T>>
std::pair<acc_t, acc_t> RowwiseMeanM2(const T* X, int64_t N) {
  constexpr int64_t kVecSize = kParityVecSize<T>;
  const int64_t n = N / kVecSize;
  const int64_t m = divup(n, kChunkSize);
  const int64_t depth = utils::CeilLog2(m);
  if (depth <= 4) {
    return RowwiseMomentsImpl<T, 4, acc_t>(X, N);
  } else if (depth <= 8) {
    return RowwiseMomentsImpl<T, 8, acc_t>(X, N);
  } else if (depth <= 16) {
    return RowwiseMomentsImpl<T, 16, acc_t>(X, N);
  } else if (depth <= 32) {
    return RowwiseMomentsImpl<T, 32, acc_t>(X, N);
  } else {
    return RowwiseMomentsImpl<T, 64, acc_t>(X, N);
  }
}

template <typename T>
std::pair<opmath_t<T>, opmath_t<T>> RowwiseMoments(const T* X, int64_t N, int64_t ddof = 0) {
  const auto [mean, m2]{RowwiseMeanM2(X, N)};
  return std::make_pair(mean, m2 / static_cast<opmath_t<T>>(N - ddof));
}

} // namespace CPU_CAPABILITY
} // namespace at::native
