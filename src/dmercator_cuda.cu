#include "../include/dmercator_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
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

  int *d_row_offsets = nullptr;
  int *d_col_indices = nullptr;
  double *d_kappa = nullptr;
  double *d_theta = nullptr;
  double *d_positions = nullptr;

  double *d_candidate_theta = nullptr;
  double *d_candidate_positions = nullptr;
  double *d_scores = nullptr;

  int candidate_capacity = 0;
  int candidate_position_capacity = 0;

  bool initialized = false;
  bool has_kappa = false;
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

    candidate_capacity = 0;
    candidate_position_capacity = 0;
    dim_plus_one = 0;
    nb_vertices = 0;
    initialized = false;
    has_kappa = false;
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
  if(!check_cuda(cudaMalloc(reinterpret_cast<void **>(&impl_->d_theta),
                            static_cast<size_t>(nb_vertices) * sizeof(double)),
                 "cudaMalloc(d_theta)", error_message))
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

} // namespace cuda
} // namespace dmercator
