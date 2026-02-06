#include "../include/dmercator_cuda.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace dmercator
{
namespace cuda
{
namespace
{

constexpr double PI = 3.141592653589793238462643383279502884197;
constexpr double NUMERICAL_ZERO = 1e-10;

template <int BLOCK_SIZE>
__device__ double block_reduce_sum(double value)
{
  __shared__ double buffer[BLOCK_SIZE];
  const int tid = threadIdx.x;
  buffer[tid] = value;
  __syncthreads();

  for(int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
  {
    if(tid < stride)
    {
      buffer[tid] += buffer[tid + stride];
    }
    __syncthreads();
  }
  return buffer[0];
}

__device__ __forceinline__ void atomic_max_nonneg_double(unsigned long long *address, double value)
{
  atomicMax(address, static_cast<unsigned long long>(__double_as_longlong(value)));
}

__device__ __forceinline__ double atomic_add_double(double *address, double value)
{
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  // Compatibility path for architectures where atomicAdd(double*) is unavailable.
  unsigned long long *address_as_ull = reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));
  } while(assumed != old);
  return __longlong_as_double(old);
#endif
}

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ double deterministic_uniform_01(unsigned long long seed,
                                                           int vertex_id,
                                                           int iteration)
{
  unsigned long long value = seed ^ (static_cast<unsigned long long>(vertex_id) * 0x9e3779b97f4a7c15ULL);
  value ^= (static_cast<unsigned long long>(iteration) + 1ULL) * 0xbf58476d1ce4e5b9ULL;
  value = splitmix64(value);
  return static_cast<double>(value >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ double wrap_angle_difference(double a1, double a2)
{
  return PI - fabs(PI - fabs(a1 - a2));
}

__device__ __forceinline__ double connection_probability_s1(double angle1,
                                                            double angle2,
                                                            double kappa1,
                                                            double kappa2,
                                                            double prefactor,
                                                            double beta)
{
  const double dtheta = wrap_angle_difference(angle1, angle2);
  const double inside = (prefactor * dtheta) / (kappa1 * kappa2);
  return 1.0 / (1.0 + pow(inside, beta));
}

__device__ __forceinline__ double connection_probability_sd(const double *positions_soa,
                                                            int dim_plus_one,
                                                            int nb_vertices,
                                                            int v1,
                                                            int v2,
                                                            double kappa1,
                                                            double kappa2,
                                                            double radius,
                                                            double mu,
                                                            double beta,
                                                            double inv_dim)
{
  double dot = 0;
  double norm1 = 0;
  double norm2 = 0;
  for(int d = 0; d < dim_plus_one; ++d)
  {
    const double p1 = positions_soa[static_cast<size_t>(d) * nb_vertices + v1];
    const double p2 = positions_soa[static_cast<size_t>(d) * nb_vertices + v2];
    dot += p1 * p2;
    norm1 += p1 * p1;
    norm2 += p2 * p2;
  }

  const double denom = sqrt(norm1) * sqrt(norm2);
  double cos_angle = dot / denom;
  if(cos_angle > 1.0)
  {
    cos_angle = 1.0;
  }
  else if(cos_angle < -1.0)
  {
    cos_angle = -1.0;
  }

  const double dtheta = (fabs(cos_angle - 1.0) < NUMERICAL_ZERO) ? 0.0 : acos(cos_angle);
  const double chi = radius * dtheta / pow(mu * kappa1 * kappa2, inv_dim);
  return 1.0 / (1.0 + pow(chi, beta));
}

template <int BLOCK_SIZE>
__global__ void score_s1_nonedge_kernel(const double *theta,
                                        const double *kappa,
                                        int nb_vertices,
                                        const double *candidate_theta,
                                        int nb_candidates,
                                        int v1,
                                        double prefactor,
                                        double beta,
                                        double *scores)
{
  const int candidate_id = blockIdx.x;
  if(candidate_id >= nb_candidates)
  {
    return;
  }

  const double angle = candidate_theta[candidate_id];
  const double kappa1 = kappa[v1];

  double local_sum = 0;
  for(int v2 = threadIdx.x; v2 < nb_vertices; v2 += BLOCK_SIZE)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double da = PI - fabs(PI - fabs(angle - theta[v2]));
    const double fraction = (prefactor * da) / (kappa1 * kappa[v2]);
    local_sum += -log1p(pow(fraction, -beta));
  }

  const double block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
  if(threadIdx.x == 0)
  {
    scores[candidate_id] = block_sum;
  }
}

template <int BLOCK_SIZE>
__global__ void score_s1_edge_kernel(const double *theta,
                                     const double *kappa,
                                     int v1,
                                     const int *row_offsets,
                                     const int *col_indices,
                                     const double *candidate_theta,
                                     int nb_candidates,
                                     double beta,
                                     double prefactor,
                                     double *scores)
{
  const int candidate_id = blockIdx.x;
  if(candidate_id >= nb_candidates)
  {
    return;
  }

  const double angle = candidate_theta[candidate_id];
  const double kappa1 = kappa[v1];

  const int begin = row_offsets[v1];
  const int end = row_offsets[v1 + 1];

  double local_sum = 0;
  for(int edge_id = begin + threadIdx.x; edge_id < end; edge_id += BLOCK_SIZE)
  {
    const int v2 = col_indices[edge_id];
    const double da = PI - fabs(PI - fabs(angle - theta[v2]));
    const double fraction = (prefactor * da) / (kappa1 * kappa[v2]);
    local_sum += -beta * log(fraction);
  }

  const double block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
  if(threadIdx.x == 0)
  {
    scores[candidate_id] += block_sum;
  }
}

template <int BLOCK_SIZE>
__global__ void score_sd_nonedge_kernel(const double *positions_soa,
                                        int dim_plus_one,
                                        int nb_vertices,
                                        const double *kappa,
                                        const double *candidate_positions_soa,
                                        int nb_candidates,
                                        int v1,
                                        double radius,
                                        double mu,
                                        double beta,
                                        double inv_dim,
                                        double *scores)
{
  const int candidate_id = blockIdx.x;
  if(candidate_id >= nb_candidates)
  {
    return;
  }

  const double kappa1 = kappa[v1];

  double local_sum = 0;
  for(int v2 = threadIdx.x; v2 < nb_vertices; v2 += BLOCK_SIZE)
  {
    if(v2 == v1)
    {
      continue;
    }

    double dot = 0;
    double norm1 = 0;
    double norm2 = 0;
    for(int d = 0; d < dim_plus_one; ++d)
    {
      const double p1 = candidate_positions_soa[d * nb_candidates + candidate_id];
      const double p2 = positions_soa[d * nb_vertices + v2];
      dot += p1 * p2;
      norm1 += p1 * p1;
      norm2 += p2 * p2;
    }

    const double denom = sqrt(norm1) * sqrt(norm2);
    double cos_angle = dot / denom;
    if(cos_angle > 1.0)
    {
      cos_angle = 1.0;
    }
    else if(cos_angle < -1.0)
    {
      cos_angle = -1.0;
    }

    const double dtheta = (fabs(cos_angle - 1.0) < NUMERICAL_ZERO) ? 0.0 : acos(cos_angle);
    const double chi = radius * dtheta / pow(mu * kappa1 * kappa[v2], inv_dim);
    local_sum += -log1p(pow(chi, -beta));
  }

  const double block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
  if(threadIdx.x == 0)
  {
    scores[candidate_id] = block_sum;
  }
}

template <int BLOCK_SIZE>
__global__ void score_sd_edge_kernel(const double *positions_soa,
                                     int dim_plus_one,
                                     int nb_vertices,
                                     const double *kappa,
                                     int v1,
                                     const int *row_offsets,
                                     const int *col_indices,
                                     const double *candidate_positions_soa,
                                     int nb_candidates,
                                     double radius,
                                     double mu,
                                     double beta,
                                     double inv_dim,
                                     double *scores)
{
  const int candidate_id = blockIdx.x;
  if(candidate_id >= nb_candidates)
  {
    return;
  }

  const double kappa1 = kappa[v1];
  const int begin = row_offsets[v1];
  const int end = row_offsets[v1 + 1];

  double local_sum = 0;
  for(int edge_id = begin + threadIdx.x; edge_id < end; edge_id += BLOCK_SIZE)
  {
    const int v2 = col_indices[edge_id];

    double dot = 0;
    double norm1 = 0;
    double norm2 = 0;
    for(int d = 0; d < dim_plus_one; ++d)
    {
      const double p1 = candidate_positions_soa[d * nb_candidates + candidate_id];
      const double p2 = positions_soa[d * nb_vertices + v2];
      dot += p1 * p2;
      norm1 += p1 * p1;
      norm2 += p2 * p2;
    }

    const double denom = sqrt(norm1) * sqrt(norm2);
    double cos_angle = dot / denom;
    if(cos_angle > 1.0)
    {
      cos_angle = 1.0;
    }
    else if(cos_angle < -1.0)
    {
      cos_angle = -1.0;
    }

    const double dtheta = (fabs(cos_angle - 1.0) < NUMERICAL_ZERO) ? 0.0 : acos(cos_angle);
    const double chi = radius * dtheta / pow(mu * kappa1 * kappa[v2], inv_dim);
    local_sum += -log1p(pow(chi, beta));
  }

  const double block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
  if(threadIdx.x == 0)
  {
    scores[candidate_id] += block_sum;
  }
}

template <int BLOCK_SIZE>
__global__ void expected_degree_s1_kernel(const double *theta,
                                          const double *kappa,
                                          int nb_vertices,
                                          double prefactor,
                                          double beta,
                                          double *expected_degree)
{
  const int v1 = blockIdx.x;
  if(v1 >= nb_vertices)
  {
    return;
  }

  const double kappa1 = kappa[v1];
  const double theta1 = theta[v1];

  double local_sum = 0;
  for(int v2 = v1 + 1 + threadIdx.x; v2 < nb_vertices; v2 += BLOCK_SIZE)
  {
    const double prob = connection_probability_s1(theta1, theta[v2], kappa1, kappa[v2], prefactor, beta);
    local_sum += prob;
    atomic_add_double(expected_degree + v2, prob);
  }

  const double block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
  if(threadIdx.x == 0)
  {
    atomic_add_double(expected_degree + v1, block_sum);
  }
}

template <int BLOCK_SIZE>
__global__ void expected_degree_sd_kernel(const double *positions_soa,
                                          int dim_plus_one,
                                          const double *kappa,
                                          int nb_vertices,
                                          double radius,
                                          double mu,
                                          double beta,
                                          double inv_dim,
                                          double *expected_degree)
{
  const int v1 = blockIdx.x;
  if(v1 >= nb_vertices)
  {
    return;
  }

  const double kappa1 = kappa[v1];

  double local_sum = 0;
  for(int v2 = v1 + 1 + threadIdx.x; v2 < nb_vertices; v2 += BLOCK_SIZE)
  {
    const double prob = connection_probability_sd(positions_soa,
                                                  dim_plus_one,
                                                  nb_vertices,
                                                  v1,
                                                  v2,
                                                  kappa1,
                                                  kappa[v2],
                                                  radius,
                                                  mu,
                                                  beta,
                                                  inv_dim);
    local_sum += prob;
    atomic_add_double(expected_degree + v2, prob);
  }

  const double block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
  if(threadIdx.x == 0)
  {
    atomic_add_double(expected_degree + v1, block_sum);
  }
}

__global__ void update_kappa_from_expected_kernel(double *kappa,
                                                  const double *expected_degree,
                                                  const int *degree,
                                                  int nb_vertices,
                                                  double convergence_threshold,
                                                  unsigned long long seed,
                                                  int iteration,
                                                  unsigned long long *max_error_bits)
{
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= nb_vertices)
  {
    return;
  }

  const double observed = static_cast<double>(degree[v]);
  const double expected = expected_degree[v];
  const double error = fabs(expected - observed);
  atomic_max_nonneg_double(max_error_bits, error);

  if(error > convergence_threshold)
  {
    const double step = deterministic_uniform_01(seed, v, iteration);
    const double updated = kappa[v] + (observed - expected) * step;
    kappa[v] = fabs(updated);
  }
}

__device__ __forceinline__ int sample_vertex_excluding(curandStatePhilox4_32_10_t *state,
                                                       int nb_vertices,
                                                       int excluded1,
                                                       int excluded2)
{
  int candidate = static_cast<int>(curand(state) % static_cast<unsigned int>(nb_vertices));
  while(candidate == excluded1 || candidate == excluded2)
  {
    candidate = static_cast<int>(curand(state) % static_cast<unsigned int>(nb_vertices));
  }
  return candidate;
}

template <int BLOCK_SIZE>
__global__ void clustering_mc_s1_kernel(const double *theta,
                                        const double *kappa,
                                        const int *degree_gt_one_vertices,
                                        int nb_degree_gt_one_vertices,
                                        int nb_vertices,
                                        double prefactor,
                                        double beta,
                                        unsigned long long seed,
                                        int nb_samples,
                                        double *out_numerator,
                                        double *out_denominator)
{
  double local_num = 0;
  double local_den = 0;

  const int sample_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(sample_id < nb_samples && nb_vertices > 2 && nb_degree_gt_one_vertices > 0)
  {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(sample_id), 0ULL, &state);

    const int center_index = static_cast<int>(curand(&state) % static_cast<unsigned int>(nb_degree_gt_one_vertices));
    const int center = degree_gt_one_vertices[center_index];
    const int v2 = sample_vertex_excluding(&state, nb_vertices, center, -1);
    const int v3 = sample_vertex_excluding(&state, nb_vertices, center, v2);

    const double p12 = connection_probability_s1(theta[center], theta[v2], kappa[center], kappa[v2], prefactor, beta);
    const double p13 = connection_probability_s1(theta[center], theta[v3], kappa[center], kappa[v3], prefactor, beta);
    const double p23 = connection_probability_s1(theta[v2], theta[v3], kappa[v2], kappa[v3], prefactor, beta);

    local_den = p12 * p13;
    local_num = local_den * p23;
  }

  const double block_num = block_reduce_sum<BLOCK_SIZE>(local_num);
  const double block_den = block_reduce_sum<BLOCK_SIZE>(local_den);
  if(threadIdx.x == 0)
  {
    atomic_add_double(out_numerator, block_num);
    atomic_add_double(out_denominator, block_den);
  }
}

template <int BLOCK_SIZE>
__global__ void clustering_mc_sd_kernel(const double *positions_soa,
                                        int dim_plus_one,
                                        const double *kappa,
                                        const int *degree_gt_one_vertices,
                                        int nb_degree_gt_one_vertices,
                                        int nb_vertices,
                                        double radius,
                                        double mu,
                                        double beta,
                                        double inv_dim,
                                        unsigned long long seed,
                                        int nb_samples,
                                        double *out_numerator,
                                        double *out_denominator)
{
  double local_num = 0;
  double local_den = 0;

  const int sample_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(sample_id < nb_samples && nb_vertices > 2 && nb_degree_gt_one_vertices > 0)
  {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(sample_id), 0ULL, &state);

    const int center_index = static_cast<int>(curand(&state) % static_cast<unsigned int>(nb_degree_gt_one_vertices));
    const int center = degree_gt_one_vertices[center_index];
    const int v2 = sample_vertex_excluding(&state, nb_vertices, center, -1);
    const int v3 = sample_vertex_excluding(&state, nb_vertices, center, v2);

    const double p12 = connection_probability_sd(positions_soa,
                                                 dim_plus_one,
                                                 nb_vertices,
                                                 center,
                                                 v2,
                                                 kappa[center],
                                                 kappa[v2],
                                                 radius,
                                                 mu,
                                                 beta,
                                                 inv_dim);
    const double p13 = connection_probability_sd(positions_soa,
                                                 dim_plus_one,
                                                 nb_vertices,
                                                 center,
                                                 v3,
                                                 kappa[center],
                                                 kappa[v3],
                                                 radius,
                                                 mu,
                                                 beta,
                                                 inv_dim);
    const double p23 = connection_probability_sd(positions_soa,
                                                 dim_plus_one,
                                                 nb_vertices,
                                                 v2,
                                                 v3,
                                                 kappa[v2],
                                                 kappa[v3],
                                                 radius,
                                                 mu,
                                                 beta,
                                                 inv_dim);

    local_den = p12 * p13;
    local_num = local_den * p23;
  }

  const double block_num = block_reduce_sum<BLOCK_SIZE>(local_num);
  const double block_den = block_reduce_sum<BLOCK_SIZE>(local_den);
  if(threadIdx.x == 0)
  {
    atomic_add_double(out_numerator, block_num);
    atomic_add_double(out_denominator, block_den);
  }
}

bool check_cuda(cudaError_t status, const std::string &context, std::string *error_message)
{
  if(status == cudaSuccess)
  {
    return true;
  }
  if(error_message)
  {
    std::ostringstream oss;
    oss << context << ": " << cudaGetErrorString(status);
    *error_message = oss.str();
  }
  return false;
}

} // namespace

struct LikelihoodBackend::Impl
{
  int nb_vertices = 0;
  int dim_plus_one = 0;
  int nb_degree_gt_one_vertices = 0;

  int *d_row_offsets = nullptr;
  int *d_col_indices = nullptr;
  int *d_degree = nullptr;
  int *d_degree_gt_one_vertices = nullptr;
  double *d_kappa = nullptr;
  double *d_theta = nullptr;
  double *d_positions = nullptr;
  double *d_expected_degree = nullptr;
  unsigned long long *d_max_error_bits = nullptr;
  double *d_clustering_numerator = nullptr;
  double *d_clustering_denominator = nullptr;

  double *d_candidate_theta = nullptr;
  double *d_candidate_positions = nullptr;
  double *d_scores = nullptr;

  int candidate_capacity = 0;
  int candidate_position_capacity = 0;

  bool initialized = false;
  bool has_kappa = false;
  bool has_degree = false;
  bool has_theta = false;
  bool has_positions = false;

  void release_all()
  {
    if(d_row_offsets)
    {
      cudaFree(d_row_offsets);
      d_row_offsets = nullptr;
    }
    if(d_col_indices)
    {
      cudaFree(d_col_indices);
      d_col_indices = nullptr;
    }
    if(d_kappa)
    {
      cudaFree(d_kappa);
      d_kappa = nullptr;
    }
    if(d_degree)
    {
      cudaFree(d_degree);
      d_degree = nullptr;
    }
    if(d_degree_gt_one_vertices)
    {
      cudaFree(d_degree_gt_one_vertices);
      d_degree_gt_one_vertices = nullptr;
    }
    if(d_theta)
    {
      cudaFree(d_theta);
      d_theta = nullptr;
    }
    if(d_positions)
    {
      cudaFree(d_positions);
      d_positions = nullptr;
    }
    if(d_candidate_theta)
    {
      cudaFree(d_candidate_theta);
      d_candidate_theta = nullptr;
    }
    if(d_candidate_positions)
    {
      cudaFree(d_candidate_positions);
      d_candidate_positions = nullptr;
    }
    if(d_scores)
    {
      cudaFree(d_scores);
      d_scores = nullptr;
    }
    if(d_expected_degree)
    {
      cudaFree(d_expected_degree);
      d_expected_degree = nullptr;
    }
    if(d_max_error_bits)
    {
      cudaFree(d_max_error_bits);
      d_max_error_bits = nullptr;
    }
    if(d_clustering_numerator)
    {
      cudaFree(d_clustering_numerator);
      d_clustering_numerator = nullptr;
    }
    if(d_clustering_denominator)
    {
      cudaFree(d_clustering_denominator);
      d_clustering_denominator = nullptr;
    }

    candidate_capacity = 0;
    candidate_position_capacity = 0;
    dim_plus_one = 0;
    nb_vertices = 0;
    nb_degree_gt_one_vertices = 0;
    initialized = false;
    has_kappa = false;
    has_degree = false;
    has_theta = false;
    has_positions = false;
  }
};

LikelihoodBackend::LikelihoodBackend() : impl_(new Impl()) {}

LikelihoodBackend::~LikelihoodBackend()
{
  if(impl_)
  {
    impl_->release_all();
    delete impl_;
    impl_ = nullptr;
  }
}

bool LikelihoodBackend::initialize(int nb_vertices,
                                   const std::vector<int> &row_offsets,
                                   const std::vector<int> &col_indices,
                                   std::string *error_message)
{
  if(!impl_)
  {
    if(error_message)
    {
      *error_message = "CUDA backend internal state is null.";
    }
    return false;
  }

  if(nb_vertices <= 0)
  {
    if(error_message)
    {
      *error_message = "Number of vertices must be positive.";
    }
    return false;
  }
  if(static_cast<int>(row_offsets.size()) != nb_vertices + 1)
  {
    if(error_message)
    {
      *error_message = "Invalid CSR row_offsets size.";
    }
    return false;
  }

  impl_->release_all();
  impl_->nb_vertices = nb_vertices;

  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_row_offsets),
                            row_offsets.size() * sizeof(int)),
                 "cudaMalloc(d_row_offsets)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_col_indices),
                            col_indices.size() * sizeof(int)),
                 "cudaMalloc(d_col_indices)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_kappa),
                            static_cast<size_t>(nb_vertices) * sizeof(double)),
                 "cudaMalloc(d_kappa)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_degree),
                            static_cast<size_t>(nb_vertices) * sizeof(int)),
                 "cudaMalloc(d_degree)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_theta),
                            static_cast<size_t>(nb_vertices) * sizeof(double)),
                 "cudaMalloc(d_theta)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_expected_degree),
                            static_cast<size_t>(nb_vertices) * sizeof(double)),
                 "cudaMalloc(d_expected_degree)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_max_error_bits),
                            sizeof(unsigned long long)),
                 "cudaMalloc(d_max_error_bits)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_clustering_numerator),
                            sizeof(double)),
                 "cudaMalloc(d_clustering_numerator)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_clustering_denominator),
                            sizeof(double)),
                 "cudaMalloc(d_clustering_denominator)", error_message))
  {
    impl_->release_all();
    return false;
  }

  if(!check_cuda(cudaMemcpy(impl_->d_row_offsets,
                            row_offsets.data(),
                            row_offsets.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(row_offsets)", error_message))
  {
    impl_->release_all();
    return false;
  }
  if(!check_cuda(cudaMemcpy(impl_->d_col_indices,
                            col_indices.data(),
                            col_indices.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(col_indices)", error_message))
  {
    impl_->release_all();
    return false;
  }

  impl_->initialized = true;
  return true;
}

bool LikelihoodBackend::is_initialized() const
{
  return impl_ && impl_->initialized;
}

int LikelihoodBackend::nb_vertices() const
{
  return impl_ ? impl_->nb_vertices : 0;
}

bool LikelihoodBackend::update_kappa(const std::vector<double> &kappa, std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(static_cast<int>(kappa.size()) != impl_->nb_vertices)
  {
    if(error_message)
    {
      *error_message = "Invalid kappa vector size.";
    }
    return false;
  }
  if(!check_cuda(cudaMemcpy(impl_->d_kappa,
                            kappa.data(),
                            static_cast<size_t>(impl_->nb_vertices) * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(kappa)", error_message))
  {
    return false;
  }
  impl_->has_kappa = true;
  return true;
}

bool LikelihoodBackend::download_kappa(std::vector<double> &kappa, std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(!impl_->has_kappa)
  {
    if(error_message)
    {
      *error_message = "Kappa buffer is not initialized on device.";
    }
    return false;
  }

  kappa.resize(impl_->nb_vertices);
  if(!check_cuda(cudaMemcpy(kappa.data(),
                            impl_->d_kappa,
                            static_cast<size_t>(impl_->nb_vertices) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(kappa D2H)", error_message))
  {
    return false;
  }
  return true;
}

bool LikelihoodBackend::update_observed_degree(const std::vector<int> &degree, std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(static_cast<int>(degree.size()) != impl_->nb_vertices)
  {
    if(error_message)
    {
      *error_message = "Invalid observed degree vector size.";
    }
    return false;
  }

  if(!check_cuda(cudaMemcpy(impl_->d_degree,
                            degree.data(),
                            static_cast<size_t>(impl_->nb_vertices) * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(observed_degree)", error_message))
  {
    return false;
  }

  std::vector<int> degree_gt_one_vertices;
  degree_gt_one_vertices.reserve(impl_->nb_vertices);
  for(int v = 0; v < impl_->nb_vertices; ++v)
  {
    if(degree[v] > 1)
    {
      degree_gt_one_vertices.push_back(v);
    }
  }
  if(degree_gt_one_vertices.empty())
  {
    for(int v = 0; v < impl_->nb_vertices; ++v)
    {
      degree_gt_one_vertices.push_back(v);
    }
  }

  if(impl_->d_degree_gt_one_vertices)
  {
    cudaFree(impl_->d_degree_gt_one_vertices);
    impl_->d_degree_gt_one_vertices = nullptr;
  }
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_degree_gt_one_vertices),
                            degree_gt_one_vertices.size() * sizeof(int)),
                 "cudaMalloc(d_degree_gt_one_vertices)", error_message))
  {
    return false;
  }
  if(!check_cuda(cudaMemcpy(impl_->d_degree_gt_one_vertices,
                            degree_gt_one_vertices.data(),
                            degree_gt_one_vertices.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(d_degree_gt_one_vertices)", error_message))
  {
    return false;
  }

  impl_->nb_degree_gt_one_vertices = static_cast<int>(degree_gt_one_vertices.size());
  impl_->has_degree = true;
  return true;
}

bool LikelihoodBackend::update_theta(const std::vector<double> &theta, std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(static_cast<int>(theta.size()) != impl_->nb_vertices)
  {
    if(error_message)
    {
      *error_message = "Invalid theta vector size.";
    }
    return false;
  }
  if(!check_cuda(cudaMemcpy(impl_->d_theta,
                            theta.data(),
                            static_cast<size_t>(impl_->nb_vertices) * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(theta)", error_message))
  {
    return false;
  }
  impl_->has_theta = true;
  return true;
}

bool LikelihoodBackend::update_theta_entry(int vertex_id,
                                           double theta_value,
                                           std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(vertex_id < 0 || vertex_id >= impl_->nb_vertices)
  {
    if(error_message)
    {
      *error_message = "Vertex id out of bounds in update_theta_entry.";
    }
    return false;
  }
  if(!check_cuda(cudaMemcpy(impl_->d_theta + vertex_id,
                            &theta_value,
                            sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(theta entry)", error_message))
  {
    return false;
  }
  impl_->has_theta = true;
  return true;
}

bool LikelihoodBackend::update_positions_soa(int dim_plus_one,
                                             const std::vector<double> &positions_soa,
                                             std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(dim_plus_one <= 0)
  {
    if(error_message)
    {
      *error_message = "dim_plus_one must be positive.";
    }
    return false;
  }

  const size_t expected_size = static_cast<size_t>(dim_plus_one) * impl_->nb_vertices;
  if(positions_soa.size() != expected_size)
  {
    if(error_message)
    {
      *error_message = "Invalid positions_soa size.";
    }
    return false;
  }

  if(!impl_->d_positions || impl_->dim_plus_one != dim_plus_one)
  {
    if(impl_->d_positions)
    {
      cudaFree(impl_->d_positions);
      impl_->d_positions = nullptr;
    }
    if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_positions),
                              expected_size * sizeof(double)),
                   "cudaMalloc(d_positions)", error_message))
    {
      return false;
    }
    impl_->dim_plus_one = dim_plus_one;
  }

  if(!check_cuda(cudaMemcpy(impl_->d_positions,
                            positions_soa.data(),
                            expected_size * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(positions_soa)", error_message))
  {
    return false;
  }
  impl_->has_positions = true;
  return true;
}

bool LikelihoodBackend::update_position_entry(int vertex_id,
                                              const std::vector<double> &position,
                                              std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(vertex_id < 0 || vertex_id >= impl_->nb_vertices)
  {
    if(error_message)
    {
      *error_message = "Vertex id out of bounds in update_position_entry.";
    }
    return false;
  }
  if(!impl_->d_positions || static_cast<int>(position.size()) != impl_->dim_plus_one)
  {
    if(error_message)
    {
      *error_message = "Positions buffer not initialized or invalid position dimension.";
    }
    return false;
  }

  for(int d = 0; d < impl_->dim_plus_one; ++d)
  {
    const double value = position[d];
    if(!check_cuda(cudaMemcpy(impl_->d_positions + static_cast<size_t>(d) * impl_->nb_vertices + vertex_id,
                              &value,
                              sizeof(double),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(position entry)", error_message))
    {
      return false;
    }
  }
  impl_->has_positions = true;
  return true;
}

bool LikelihoodBackend::score_candidates_s1(int v1,
                                            double prefactor,
                                            double beta,
                                            const std::vector<double> &candidate_theta,
                                            std::vector<double> &out_scores,
                                            std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(!impl_->has_kappa || !impl_->has_theta)
  {
    if(error_message)
    {
      *error_message = "Kappa/theta buffers are not initialized on device.";
    }
    return false;
  }
  if(v1 < 0 || v1 >= impl_->nb_vertices)
  {
    if(error_message)
    {
      *error_message = "Vertex id out of bounds in score_candidates_s1.";
    }
    return false;
  }
  if(candidate_theta.empty())
  {
    out_scores.clear();
    return true;
  }

  const int nb_candidates = static_cast<int>(candidate_theta.size());
  if(nb_candidates > impl_->candidate_capacity)
  {
    if(impl_->d_candidate_theta)
    {
      cudaFree(impl_->d_candidate_theta);
      impl_->d_candidate_theta = nullptr;
    }
    if(impl_->d_scores)
    {
      cudaFree(impl_->d_scores);
      impl_->d_scores = nullptr;
    }
    if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_candidate_theta),
                              candidate_theta.size() * sizeof(double)),
                   "cudaMalloc(d_candidate_theta)", error_message))
    {
      return false;
    }
    if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_scores),
                              candidate_theta.size() * sizeof(double)),
                   "cudaMalloc(d_scores)", error_message))
    {
      return false;
    }
    impl_->candidate_capacity = nb_candidates;
  }

  if(!check_cuda(cudaMemcpy(impl_->d_candidate_theta,
                            candidate_theta.data(),
                            candidate_theta.size() * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(candidate_theta)", error_message))
  {
    return false;
  }

  constexpr int BLOCK_SIZE = 256;
  score_s1_nonedge_kernel<BLOCK_SIZE><<<nb_candidates, BLOCK_SIZE>>>(impl_->d_theta,
                                                                      impl_->d_kappa,
                                                                      impl_->nb_vertices,
                                                                      impl_->d_candidate_theta,
                                                                      nb_candidates,
                                                                      v1,
                                                                      prefactor,
                                                                      beta,
                                                                      impl_->d_scores);
  if(!check_cuda(cudaGetLastError(), "score_s1_nonedge_kernel launch", error_message))
  {
    return false;
  }

  score_s1_edge_kernel<BLOCK_SIZE><<<nb_candidates, BLOCK_SIZE>>>(impl_->d_theta,
                                                                   impl_->d_kappa,
                                                                   v1,
                                                                   impl_->d_row_offsets,
                                                                   impl_->d_col_indices,
                                                                   impl_->d_candidate_theta,
                                                                   nb_candidates,
                                                                   beta,
                                                                   prefactor,
                                                                   impl_->d_scores);
  if(!check_cuda(cudaGetLastError(), "score_s1_edge_kernel launch", error_message))
  {
    return false;
  }

  out_scores.resize(nb_candidates);
  if(!check_cuda(cudaMemcpy(out_scores.data(),
                            impl_->d_scores,
                            out_scores.size() * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(scores s1)", error_message))
  {
    return false;
  }

  return true;
}

bool LikelihoodBackend::score_candidates_sd(int dim,
                                            int v1,
                                            double radius,
                                            double mu,
                                            double beta,
                                            const std::vector<double> &candidate_positions_soa,
                                            std::vector<double> &out_scores,
                                            std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(!impl_->has_kappa || !impl_->has_positions)
  {
    if(error_message)
    {
      *error_message = "Kappa/positions buffers are not initialized on device.";
    }
    return false;
  }
  if(dim <= 0)
  {
    if(error_message)
    {
      *error_message = "Invalid dimension in score_candidates_sd.";
    }
    return false;
  }
  if(v1 < 0 || v1 >= impl_->nb_vertices)
  {
    if(error_message)
    {
      *error_message = "Vertex id out of bounds in score_candidates_sd.";
    }
    return false;
  }

  const int dim_plus_one = dim + 1;
  if(impl_->dim_plus_one != dim_plus_one)
  {
    if(error_message)
    {
      *error_message = "Device positions dimension mismatch.";
    }
    return false;
  }
  if(candidate_positions_soa.empty())
  {
    out_scores.clear();
    return true;
  }
  if(candidate_positions_soa.size() % static_cast<size_t>(dim_plus_one) != 0)
  {
    if(error_message)
    {
      *error_message = "Invalid candidate_positions_soa size.";
    }
    return false;
  }

  const int nb_candidates = static_cast<int>(candidate_positions_soa.size() / dim_plus_one);
  const int candidate_position_size = static_cast<int>(candidate_positions_soa.size());

  if(nb_candidates > impl_->candidate_capacity)
  {
    if(impl_->d_scores)
    {
      cudaFree(impl_->d_scores);
      impl_->d_scores = nullptr;
    }
    if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_scores),
                              static_cast<size_t>(nb_candidates) * sizeof(double)),
                   "cudaMalloc(d_scores sd)", error_message))
    {
      return false;
    }
    impl_->candidate_capacity = nb_candidates;
  }

  if(candidate_position_size > impl_->candidate_position_capacity)
  {
    if(impl_->d_candidate_positions)
    {
      cudaFree(impl_->d_candidate_positions);
      impl_->d_candidate_positions = nullptr;
    }
    if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_candidate_positions),
                              static_cast<size_t>(candidate_position_size) * sizeof(double)),
                   "cudaMalloc(d_candidate_positions)", error_message))
    {
      return false;
    }
    impl_->candidate_position_capacity = candidate_position_size;
  }

  if(!check_cuda(cudaMemcpy(impl_->d_candidate_positions,
                            candidate_positions_soa.data(),
                            candidate_positions_soa.size() * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(candidate_positions_soa)", error_message))
  {
    return false;
  }

  constexpr int BLOCK_SIZE = 256;
  score_sd_nonedge_kernel<BLOCK_SIZE><<<nb_candidates, BLOCK_SIZE>>>(impl_->d_positions,
                                                                      dim_plus_one,
                                                                      impl_->nb_vertices,
                                                                      impl_->d_kappa,
                                                                      impl_->d_candidate_positions,
                                                                      nb_candidates,
                                                                      v1,
                                                                      radius,
                                                                      mu,
                                                                      beta,
                                                                      1.0 / dim,
                                                                      impl_->d_scores);
  if(!check_cuda(cudaGetLastError(), "score_sd_nonedge_kernel launch", error_message))
  {
    return false;
  }

  score_sd_edge_kernel<BLOCK_SIZE><<<nb_candidates, BLOCK_SIZE>>>(impl_->d_positions,
                                                                   dim_plus_one,
                                                                   impl_->nb_vertices,
                                                                   impl_->d_kappa,
                                                                   v1,
                                                                   impl_->d_row_offsets,
                                                                   impl_->d_col_indices,
                                                                   impl_->d_candidate_positions,
                                                                   nb_candidates,
                                                                   radius,
                                                                   mu,
                                                                   beta,
                                                                   1.0 / dim,
                                                                   impl_->d_scores);
  if(!check_cuda(cudaGetLastError(), "score_sd_edge_kernel launch", error_message))
  {
    return false;
  }

  out_scores.resize(nb_candidates);
  if(!check_cuda(cudaMemcpy(out_scores.data(),
                            impl_->d_scores,
                            out_scores.size() * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(scores sd)", error_message))
  {
    return false;
  }

  return true;
}

bool LikelihoodBackend::run_kappa_iteration_s1(double prefactor,
                                               double beta,
                                               double convergence_threshold,
                                               unsigned long long seed,
                                               int iteration,
                                               double *out_max_abs_error,
                                               std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(!impl_->has_kappa || !impl_->has_theta || !impl_->has_degree)
  {
    if(error_message)
    {
      *error_message = "Kappa/theta/degree buffers are not initialized on device.";
    }
    return false;
  }

  if(!check_cuda(cudaMemset(impl_->d_expected_degree, 0, static_cast<size_t>(impl_->nb_vertices) * sizeof(double)),
                 "cudaMemset(d_expected_degree)", error_message))
  {
    return false;
  }

  constexpr int BLOCK_SIZE = 256;
  expected_degree_s1_kernel<BLOCK_SIZE><<<impl_->nb_vertices, BLOCK_SIZE>>>(impl_->d_theta,
                                                                             impl_->d_kappa,
                                                                             impl_->nb_vertices,
                                                                             prefactor,
                                                                             beta,
                                                                             impl_->d_expected_degree);
  if(!check_cuda(cudaGetLastError(), "expected_degree_s1_kernel launch", error_message))
  {
    return false;
  }

  if(!check_cuda(cudaMemset(impl_->d_max_error_bits, 0, sizeof(unsigned long long)),
                 "cudaMemset(d_max_error_bits)", error_message))
  {
    return false;
  }

  const int blocks = (impl_->nb_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
  update_kappa_from_expected_kernel<<<blocks, BLOCK_SIZE>>>(impl_->d_kappa,
                                                             impl_->d_expected_degree,
                                                             impl_->d_degree,
                                                             impl_->nb_vertices,
                                                             convergence_threshold,
                                                             seed,
                                                             iteration,
                                                             impl_->d_max_error_bits);
  if(!check_cuda(cudaGetLastError(), "update_kappa_from_expected_kernel s1 launch", error_message))
  {
    return false;
  }

  unsigned long long max_error_bits = 0;
  if(!check_cuda(cudaMemcpy(&max_error_bits,
                            impl_->d_max_error_bits,
                            sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(max_error_bits s1)", error_message))
  {
    return false;
  }

  if(out_max_abs_error)
  {
    double value = 0.0;
    std::memcpy(&value, &max_error_bits, sizeof(double));
    *out_max_abs_error = value;
  }
  return true;
}

bool LikelihoodBackend::run_kappa_iteration_sd(int dim,
                                               double radius,
                                               double mu,
                                               double beta,
                                               double convergence_threshold,
                                               unsigned long long seed,
                                               int iteration,
                                               double *out_max_abs_error,
                                               std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(dim <= 0)
  {
    if(error_message)
    {
      *error_message = "Invalid dimension in run_kappa_iteration_sd.";
    }
    return false;
  }
  if(!impl_->has_kappa || !impl_->has_positions || !impl_->has_degree)
  {
    if(error_message)
    {
      *error_message = "Kappa/positions/degree buffers are not initialized on device.";
    }
    return false;
  }
  if(impl_->dim_plus_one != dim + 1)
  {
    if(error_message)
    {
      *error_message = "Device positions dimension mismatch.";
    }
    return false;
  }

  if(!check_cuda(cudaMemset(impl_->d_expected_degree, 0, static_cast<size_t>(impl_->nb_vertices) * sizeof(double)),
                 "cudaMemset(d_expected_degree sd)", error_message))
  {
    return false;
  }

  constexpr int BLOCK_SIZE = 256;
  expected_degree_sd_kernel<BLOCK_SIZE><<<impl_->nb_vertices, BLOCK_SIZE>>>(impl_->d_positions,
                                                                             impl_->dim_plus_one,
                                                                             impl_->d_kappa,
                                                                             impl_->nb_vertices,
                                                                             radius,
                                                                             mu,
                                                                             beta,
                                                                             1.0 / dim,
                                                                             impl_->d_expected_degree);
  if(!check_cuda(cudaGetLastError(), "expected_degree_sd_kernel launch", error_message))
  {
    return false;
  }

  if(!check_cuda(cudaMemset(impl_->d_max_error_bits, 0, sizeof(unsigned long long)),
                 "cudaMemset(d_max_error_bits sd)", error_message))
  {
    return false;
  }

  const int blocks = (impl_->nb_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
  update_kappa_from_expected_kernel<<<blocks, BLOCK_SIZE>>>(impl_->d_kappa,
                                                             impl_->d_expected_degree,
                                                             impl_->d_degree,
                                                             impl_->nb_vertices,
                                                             convergence_threshold,
                                                             seed,
                                                             iteration,
                                                             impl_->d_max_error_bits);
  if(!check_cuda(cudaGetLastError(), "update_kappa_from_expected_kernel sd launch", error_message))
  {
    return false;
  }

  unsigned long long max_error_bits = 0;
  if(!check_cuda(cudaMemcpy(&max_error_bits,
                            impl_->d_max_error_bits,
                            sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(max_error_bits sd)", error_message))
  {
    return false;
  }

  if(out_max_abs_error)
  {
    double value = 0.0;
    std::memcpy(&value, &max_error_bits, sizeof(double));
    *out_max_abs_error = value;
  }
  return true;
}

bool LikelihoodBackend::estimate_mean_clustering_s1(double prefactor,
                                                    double beta,
                                                    unsigned long long seed,
                                                    int nb_samples,
                                                    double *out_mean_clustering,
                                                    std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(!impl_->has_kappa || !impl_->has_theta || !impl_->has_degree)
  {
    if(error_message)
    {
      *error_message = "Kappa/theta/degree buffers are not initialized on device.";
    }
    return false;
  }
  if(nb_samples <= 0)
  {
    if(error_message)
    {
      *error_message = "Number of Monte Carlo samples must be positive.";
    }
    return false;
  }

  if(!check_cuda(cudaMemset(impl_->d_clustering_numerator, 0, sizeof(double)),
                 "cudaMemset(d_clustering_numerator)", error_message))
  {
    return false;
  }
  if(!check_cuda(cudaMemset(impl_->d_clustering_denominator, 0, sizeof(double)),
                 "cudaMemset(d_clustering_denominator)", error_message))
  {
    return false;
  }

  constexpr int BLOCK_SIZE = 256;
  const int blocks = (nb_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
  clustering_mc_s1_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(impl_->d_theta,
                                                               impl_->d_kappa,
                                                               impl_->d_degree_gt_one_vertices,
                                                               impl_->nb_degree_gt_one_vertices,
                                                               impl_->nb_vertices,
                                                               prefactor,
                                                               beta,
                                                               seed,
                                                               nb_samples,
                                                               impl_->d_clustering_numerator,
                                                               impl_->d_clustering_denominator);
  if(!check_cuda(cudaGetLastError(), "clustering_mc_s1_kernel launch", error_message))
  {
    return false;
  }

  double numerator = 0;
  double denominator = 0;
  if(!check_cuda(cudaMemcpy(&numerator,
                            impl_->d_clustering_numerator,
                            sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(clustering numerator s1)", error_message))
  {
    return false;
  }
  if(!check_cuda(cudaMemcpy(&denominator,
                            impl_->d_clustering_denominator,
                            sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(clustering denominator s1)", error_message))
  {
    return false;
  }

  if(out_mean_clustering)
  {
    *out_mean_clustering = (denominator > NUMERICAL_ZERO) ? (numerator / denominator) : 0.0;
  }
  return true;
}

bool LikelihoodBackend::estimate_mean_clustering_sd(int dim,
                                                    double radius,
                                                    double mu,
                                                    double beta,
                                                    unsigned long long seed,
                                                    int nb_samples,
                                                    double *out_mean_clustering,
                                                    std::string *error_message)
{
  if(!is_initialized())
  {
    if(error_message)
    {
      *error_message = "CUDA backend is not initialized.";
    }
    return false;
  }
  if(dim <= 0)
  {
    if(error_message)
    {
      *error_message = "Invalid dimension in estimate_mean_clustering_sd.";
    }
    return false;
  }
  if(!impl_->has_kappa || !impl_->has_positions || !impl_->has_degree)
  {
    if(error_message)
    {
      *error_message = "Kappa/positions/degree buffers are not initialized on device.";
    }
    return false;
  }
  if(impl_->dim_plus_one != dim + 1)
  {
    if(error_message)
    {
      *error_message = "Device positions dimension mismatch.";
    }
    return false;
  }
  if(nb_samples <= 0)
  {
    if(error_message)
    {
      *error_message = "Number of Monte Carlo samples must be positive.";
    }
    return false;
  }

  if(!check_cuda(cudaMemset(impl_->d_clustering_numerator, 0, sizeof(double)),
                 "cudaMemset(d_clustering_numerator sd)", error_message))
  {
    return false;
  }
  if(!check_cuda(cudaMemset(impl_->d_clustering_denominator, 0, sizeof(double)),
                 "cudaMemset(d_clustering_denominator sd)", error_message))
  {
    return false;
  }

  constexpr int BLOCK_SIZE = 256;
  const int blocks = (nb_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
  clustering_mc_sd_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(impl_->d_positions,
                                                               impl_->dim_plus_one,
                                                               impl_->d_kappa,
                                                               impl_->d_degree_gt_one_vertices,
                                                               impl_->nb_degree_gt_one_vertices,
                                                               impl_->nb_vertices,
                                                               radius,
                                                               mu,
                                                               beta,
                                                               1.0 / dim,
                                                               seed,
                                                               nb_samples,
                                                               impl_->d_clustering_numerator,
                                                               impl_->d_clustering_denominator);
  if(!check_cuda(cudaGetLastError(), "clustering_mc_sd_kernel launch", error_message))
  {
    return false;
  }

  double numerator = 0;
  double denominator = 0;
  if(!check_cuda(cudaMemcpy(&numerator,
                            impl_->d_clustering_numerator,
                            sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(clustering numerator sd)", error_message))
  {
    return false;
  }
  if(!check_cuda(cudaMemcpy(&denominator,
                            impl_->d_clustering_denominator,
                            sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(clustering denominator sd)", error_message))
  {
    return false;
  }

  if(out_mean_clustering)
  {
    *out_mean_clustering = (denominator > NUMERICAL_ZERO) ? (numerator / denominator) : 0.0;
  }
  return true;
}

} // namespace cuda
} // namespace dmercator
