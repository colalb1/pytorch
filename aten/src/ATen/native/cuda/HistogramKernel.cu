#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Histogram.h>

#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace at::native {

namespace {

// Binning must stay bit-for-bit consistent with the CPU reference in
// native/cpu/HistogramKernel.cpp; accumulation uses privatized shared-memory
// atomics in the style of native/cuda/SummaryOps.cu.
enum BIN_SELECTION_ALGORITHM {
    LINEAR_INTERPOLATION,
    LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
    BINARY_SEARCH,
};

// Equivalent to std::upper_bound(data, data + count, val) - data.
template <typename scalar_t>
__device__ __forceinline__ int64_t upper_bound_offset(
    const scalar_t* data, int64_t count, scalar_t val) {
  int64_t lo = 0;
  while (count > 0) {
    const int64_t step = count >> 1;
    const int64_t mid = lo + step;
    if (!(val < data[mid])) {
      lo = mid + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return lo;
}

// meta packs three length-D int64 arrays: [0, D) bin-edge counts, [D, 2D) each
// dimension's offset into the concatenated bin_edges, [2D, 3D) flat-hist strides.
template <typename scalar_t, BIN_SELECTION_ALGORITHM algorithm>
C10_LAUNCH_BOUNDS_1(at::cuda::getApplyBlockSize())
__global__ void histogramdd_cuda_kernel(
    scalar_t* __restrict__ hist,
    int64_t total_bins,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bin_edges,
    const int64_t* __restrict__ meta,
    int64_t M,
    int64_t D,
    bool use_shared) {
  extern __shared__ unsigned char smem_raw[];
  scalar_t* smem = use_shared ? reinterpret_cast<scalar_t*>(smem_raw) : nullptr;
  scalar_t* acc = use_shared ? smem : hist;

  if (use_shared) {
    for (int64_t b = threadIdx.x; b < total_bins; b += blockDim.x) {
      smem[b] = scalar_t(0);
    }
    __syncthreads();
  }

  const int64_t* num_edges = meta;
  const int64_t* offsets = meta + D;
  const int64_t* hist_strides = meta + 2 * D;

  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < M;
       i += gridDim.x * blockDim.x) {
    bool skip_elt = false;
    int64_t hist_index = 0;

    for (int64_t dim = 0; dim < D; dim++) {
      const scalar_t elt = input[i * D + dim];
      const scalar_t* edges = bin_edges + offsets[dim];
      const int64_t num_bin_edges = num_edges[dim];
      const scalar_t leftmost_edge = edges[0];
      const scalar_t rightmost_edge = edges[num_bin_edges - 1];

      if (!(elt >= leftmost_edge && elt <= rightmost_edge)) {
        skip_elt = true;
        break;
      }

      int64_t pos = -1;
      if (algorithm == BINARY_SEARCH) {
        pos = upper_bound_offset(edges, num_bin_edges, elt) - 1;
      } else {
        pos = static_cast<int64_t>((elt - leftmost_edge) * (num_bin_edges - 1)
                / (rightmost_edge - leftmost_edge));
        if (algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
          const int64_t pos_min = pos - 1 > 0 ? pos - 1 : 0;
          const int64_t pos_max = pos + 2 < num_bin_edges ? pos + 2 : num_bin_edges;
          pos = pos_min + upper_bound_offset(edges + pos_min, pos_max - pos_min, elt) - 1;
        }
      }

      // Unlike other bins, the rightmost bin includes its right boundary
      if (pos == num_bin_edges - 1) {
        pos -= 1;
      }

      hist_index += hist_strides[dim] * pos;
    }

    if (!skip_elt) {
      const scalar_t wt = weight != nullptr ? weight[i] : scalar_t(1);
      gpuAtomicAddNoReturn(&acc[hist_index], wt);
    }
  }

  if (use_shared) {
    __syncthreads();
    for (int64_t b = threadIdx.x; b < total_bins; b += blockDim.x) {
      gpuAtomicAddNoReturn(&hist[b], smem[b]);
    }
  }
}

// hist, input (shape (M, D)) and bin_edges are all contiguous here.
template <typename scalar_t, BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_cuda_contiguous(Tensor& hist, const TensorList& bin_edges,
        const Tensor& input, const std::optional<Tensor>& weight) {
  const int64_t D = input.size(1);
  if (D == 0) {
    return;
  }

  const int64_t M = input.size(0);
  const int64_t total_bins = hist.numel();
  if (M == 0) {
    return;
  }

  const Tensor bin_edges_flat = at::cat(bin_edges, 0);

  Tensor meta_cpu = at::empty({3 * D}, input.options().dtype(at::kLong).device(at::kCPU));
  int64_t* meta_data = meta_cpu.data_ptr<int64_t>();
  int64_t offset = 0;
  for (const auto dim : c10::irange(D)) {
    const int64_t num_bin_edges = bin_edges[dim].numel();
    meta_data[dim] = num_bin_edges;
    meta_data[D + dim] = offset;
    meta_data[2 * D + dim] = hist.stride(dim);
    offset += num_bin_edges;
  }

  const Tensor meta = meta_cpu.to(hist.device());

  auto props = at::cuda::getCurrentDeviceProperties();
  const size_t shared_bytes = total_bins * sizeof(scalar_t) + 8;  // 8 guard bytes
  const bool use_shared = shared_bytes < static_cast<size_t>(props->sharedMemPerBlock);

  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;

  const auto cur_device = at::cuda::current_device();
  TORCH_INTERNAL_ASSERT(cur_device != -1);

  const bool got_grid = at::cuda::getApplyGrid(M, grid, cur_device);
  TORCH_INTERNAL_ASSERT(got_grid);

  if (use_shared) {
    // Cap grid.x so the shared-histogram flush cost stays bounded (see SummaryOps.cu).
    constexpr size_t gmem_to_smem_ratio = 8;
    unsigned int optimal_grid = at::ceil_div<size_t>(
        gmem_to_smem_ratio * static_cast<size_t>(M),
        static_cast<size_t>(total_bins) * props->multiProcessorCount);
    if (optimal_grid < static_cast<unsigned int>(props->multiProcessorCount)) {
      optimal_grid = 1 + static_cast<unsigned int>(
          std::sqrt(gmem_to_smem_ratio * static_cast<double>(M) / total_bins));
    }
    const size_t optimal_steps = at::ceil_div<size_t>(
        static_cast<size_t>(M), static_cast<size_t>(optimal_grid) * block.x);
    optimal_grid = at::ceil_div<size_t>(
        static_cast<size_t>(M), optimal_steps * block.x);
    grid.x = std::min(grid.x, optimal_grid);
  }

  const scalar_t* weight_data = weight.has_value()
          ? weight.value().const_data_ptr<scalar_t>() : nullptr;
  const size_t launch_smem = use_shared ? shared_bytes : 0;

  histogramdd_cuda_kernel<scalar_t, algorithm>
      <<<grid, block, launch_smem, at::cuda::getCurrentCUDAStream()>>>(
          hist.mutable_data_ptr<scalar_t>(),
          total_bins,
          input.const_data_ptr<scalar_t>(),
          weight_data,
          bin_edges_flat.const_data_ptr<scalar_t>(),
          meta.const_data_ptr<int64_t>(),
          M,
          D,
          use_shared);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Mirrors histogramdd_out_cpu_template. Accumulates into a contiguous buffer so a
// non-contiguous out= histogram is handled by the final copy_.
template <BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_out_cuda_template(const Tensor& self, const std::optional<Tensor>& weight,
        bool density, Tensor& hist, const TensorList& bin_edges) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of floating point atomicAdd usage
  globalContext().alertNotDeterministic("histogramdd_cuda");

  Tensor work = hist.is_contiguous() ? hist : at::empty(hist.sizes(), hist.options());
  work.zero_();

  const int64_t N = self.size(-1);
  const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
          static_cast<int64_t>(1), std::multiplies<int64_t>());

  const Tensor reshaped_input = self.reshape({M, N}).contiguous();
  const auto reshaped_weight = weight.has_value()
          ? std::optional<Tensor>(weight.value().reshape({M}).contiguous())
          : std::optional<Tensor>();

  std::vector<Tensor> bin_edges_contig(bin_edges.size());
  for (const auto dim : c10::irange(bin_edges_contig.size())) {
    bin_edges_contig[dim] = bin_edges[dim].contiguous();
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "histogram_cuda", [&]() {
    histogramdd_cuda_contiguous<scalar_t, algorithm>(
            work, bin_edges_contig, reshaped_input, reshaped_weight);
  });

  // density: divide each bin by the total weight and by the bin's volume.
  if (density) {
    const auto hist_sum = work.sum().item();
    work.div_(hist_sum);

    for (const auto dim : c10::irange(N)) {
      const auto bin_lengths = bin_edges[dim].diff();

      // Align bin_lengths with hist's dim-th axis.
      std::vector<int64_t> shape(N, 1);
      shape[dim] = bin_lengths.numel();

      work.div_(bin_lengths.reshape(shape));
    }
  }

  if (!work.is_same(hist)) {
    hist.copy_(work);
  }
}

void histogramdd_kernel_impl(const Tensor& self, const std::optional<Tensor>& weight, bool density,
        Tensor& hist, const TensorList& bin_edges) {
  histogramdd_out_cuda_template<BINARY_SEARCH>(self, weight, density, hist, bin_edges);
}

void histogramdd_linear_kernel_impl(const Tensor& self, const std::optional<Tensor>& weight,
        bool density, Tensor& hist, const TensorList& bin_edges, bool local_search) {
  if (local_search) {
    histogramdd_out_cuda_template<LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH>(
            self, weight, density, hist, bin_edges);
  } else {
    histogramdd_out_cuda_template<LINEAR_INTERPOLATION>(
            self, weight, density, hist, bin_edges);
  }
}

template <typename scalar_t>
void infer_bin_edges_from_input_cuda(const Tensor& input, const int64_t N,
        std::vector<double>& leftmost_edges, std::vector<double>& rightmost_edges) {
  auto [min, max] = at::aminmax(input, 0);

  const Tensor min_cpu = min.contiguous().cpu();
  const Tensor max_cpu = max.contiguous().cpu();

  const scalar_t* min_data = min_cpu.const_data_ptr<scalar_t>();
  std::copy(min_data, min_data + N, leftmost_edges.begin());

  const scalar_t* max_data = max_cpu.const_data_ptr<scalar_t>();
  std::copy(max_data, max_data + N, rightmost_edges.begin());
}

void histogram_select_outer_bin_edges_impl(const Tensor& input, const int64_t N,
        std::vector<double>& leftmost_edges, std::vector<double>& rightmost_edges) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "histogramdd", [&]() {
    infer_bin_edges_from_input_cuda<scalar_t>(input, N, leftmost_edges, rightmost_edges);
  });
}

} // namespace

REGISTER_CUDA_DISPATCH(histogramdd_stub, &histogramdd_kernel_impl)
REGISTER_CUDA_DISPATCH(histogramdd_linear_stub, &histogramdd_linear_kernel_impl)
REGISTER_CUDA_DISPATCH(histogram_select_outer_bin_edges_stub, &histogram_select_outer_bin_edges_impl)

} // namespace at::native
