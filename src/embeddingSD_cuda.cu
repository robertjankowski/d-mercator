#include "../include/hyperbolic_loglikelihood_cuda.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace embeddingSD_cuda {

namespace {

constexpr double kNumericalZeroDevice = kNumericalZero;
constexpr double kPiDevice = 3.14159265358979323846;

__device__ double compute_angle_d_vectors_device(const double *v1,
                                                 const double *v2,
                                                 int dim_plus_one) {
  double dot = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for (int i = 0; i < dim_plus_one; ++i) {
    dot += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  const double result = dot / (norm1 * norm2);
  if (fabs(result - 1.0) < kNumericalZeroDevice) {
    return 0.0;
  }
  return acos(result);
}

__device__ double compute_loglikelihood_device(bool neighbors,
                                               double dtheta,
                                               double kappa1,
                                               double kappa2,
                                               int dim,
                                               double mu,
                                               double beta,
                                               double radius) {
  const double chi = radius * dtheta / pow(mu * kappa1 * kappa2, 1.0 / dim);
  const double prob = 1.0 / (1.0 + pow(chi, beta));
  return neighbors ? log(prob) : log(1.0 - prob);
}

__device__ double compute_loglikelihood_device_1d(bool neighbors,
                                                  double theta1,
                                                  double theta2,
                                                  double kappa1,
                                                  double kappa2,
                                                  int nb_vertices,
                                                  double mu,
                                                  double beta) {
  const double da = kPiDevice - fabs(kPiDevice - fabs(theta1 - theta2));
  const double fraction = (nb_vertices * da) / (2.0 * kPiDevice * mu * kappa1 * kappa2);
  if (neighbors) {
    return -beta * log(fraction);
  }
  return -log(1.0 + pow(fraction, -beta));
}

__global__ void loglikelihood_all_kernel(const double *positions,
                                         const double *kappas,
                                         int nb_vertices,
                                         int dim_plus_one,
                                         int dim,
                                         double mu,
                                         double beta,
                                         double radius,
                                         const double *pos1,
                                         int v1,
                                         double kappa1,
                                         double *output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nb_vertices) {
    return;
  }
  if (idx == v1) {
    output[idx] = 0.0;
    return;
  }
  const double *pos2 = positions + (idx * dim_plus_one);
  const double dtheta = compute_angle_d_vectors_device(pos1, pos2, dim_plus_one);
  output[idx] = compute_loglikelihood_device(false, dtheta, kappa1, kappas[idx], dim, mu, beta, radius);
}

__global__ void loglikelihood_neighbors_kernel(const double *positions,
                                               const double *kappas,
                                               const int *neighbors,
                                               int neighbor_count,
                                               int dim_plus_one,
                                               int dim,
                                               double mu,
                                               double beta,
                                               double radius,
                                               const double *pos1,
                                               double kappa1,
                                               double *output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= neighbor_count) {
    return;
  }
  const int v2 = neighbors[idx];
  const double *pos2 = positions + (v2 * dim_plus_one);
  const double dtheta = compute_angle_d_vectors_device(pos1, pos2, dim_plus_one);
  output[idx] = compute_loglikelihood_device(true, dtheta, kappa1, kappas[v2], dim, mu, beta, radius);
}

__global__ void loglikelihood_all_kernel_1d(const double *thetas,
                                            const double *kappas,
                                            int nb_vertices,
                                            double mu,
                                            double beta,
                                            double theta1,
                                            int v1,
                                            double kappa1,
                                            double *output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nb_vertices) {
    return;
  }
  if (idx == v1) {
    output[idx] = 0.0;
    return;
  }
  output[idx] = compute_loglikelihood_device_1d(false, theta1, thetas[idx], kappa1, kappas[idx], nb_vertices, mu, beta);
}

__global__ void loglikelihood_neighbors_kernel_1d(const double *thetas,
                                                  const double *kappas,
                                                  const int *neighbors,
                                                  int neighbor_count,
                                                  int nb_vertices,
                                                  double mu,
                                                  double beta,
                                                  double theta1,
                                                  double kappa1,
                                                  double *output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= neighbor_count) {
    return;
  }
  const int v2 = neighbors[idx];
  output[idx] = compute_loglikelihood_device_1d(true, theta1, thetas[v2], kappa1, kappas[v2], nb_vertices, mu, beta);
}

bool cuda_call_succeeded(cudaError_t status) {
  return status == cudaSuccess;
}

} // namespace

LogLikelihoodWorkspace::LogLikelihoodWorkspace(int dim, int nb_vertices)
    : dim_(dim), nb_vertices_(nb_vertices), dim_plus_one_(dim + 1) {
  const std::size_t positions_count = static_cast<std::size_t>(nb_vertices_) * dim_plus_one_;
  positions_.resize(positions_count);
  kappas_.resize(nb_vertices_);

  if (!cuda_available()) {
    cuda_enabled_ = false;
    return;
  }

  const std::size_t positions_bytes = positions_count * sizeof(double);
  const std::size_t kappas_bytes = static_cast<std::size_t>(nb_vertices_) * sizeof(double);
  const std::size_t output_bytes = static_cast<std::size_t>(nb_vertices_) * sizeof(double);
  const std::size_t neighbors_bytes = static_cast<std::size_t>(nb_vertices_) * sizeof(int);
  const std::size_t pos1_bytes = static_cast<std::size_t>(dim_plus_one_) * sizeof(double);

  cuda_enabled_ = cuda_call_succeeded(cudaMalloc(&d_positions_, positions_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_kappas_, kappas_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_pos1_, pos1_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_output_all_, output_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_output_neighbors_, output_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_neighbors_, neighbors_bytes));

  if (!cuda_enabled_) {
    if (d_positions_) {
      cudaFree(d_positions_);
    }
    if (d_kappas_) {
      cudaFree(d_kappas_);
    }
    if (d_pos1_) {
      cudaFree(d_pos1_);
    }
    if (d_output_all_) {
      cudaFree(d_output_all_);
    }
    if (d_output_neighbors_) {
      cudaFree(d_output_neighbors_);
    }
    if (d_neighbors_) {
      cudaFree(d_neighbors_);
    }
    d_positions_ = nullptr;
    d_kappas_ = nullptr;
    d_pos1_ = nullptr;
    d_output_all_ = nullptr;
    d_output_neighbors_ = nullptr;
    d_neighbors_ = nullptr;
  }
}

LogLikelihoodWorkspace::~LogLikelihoodWorkspace() {
  if (d_positions_) {
    cudaFree(d_positions_);
  }
  if (d_kappas_) {
    cudaFree(d_kappas_);
  }
  if (d_pos1_) {
    cudaFree(d_pos1_);
  }
  if (d_output_all_) {
    cudaFree(d_output_all_);
  }
  if (d_output_neighbors_) {
    cudaFree(d_output_neighbors_);
  }
  if (d_neighbors_) {
    cudaFree(d_neighbors_);
  }
}

bool LogLikelihoodWorkspace::is_cuda_enabled() const {
  return cuda_enabled_;
}

void LogLikelihoodWorkspace::set_positions(const double *positions, std::size_t count) {
  positions_.assign(positions, positions + count);
  if (!cuda_enabled_) {
    return;
  }
  cudaMemcpy(d_positions_, positions_.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void LogLikelihoodWorkspace::set_kappas(const double *kappas, std::size_t count) {
  kappas_.assign(kappas, kappas + count);
  if (!cuda_enabled_) {
    return;
  }
  cudaMemcpy(d_kappas_, kappas_.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void LogLikelihoodWorkspace::update_position(int index, const double *position) {
  const std::size_t offset = static_cast<std::size_t>(index) * dim_plus_one_;
  for (int i = 0; i < dim_plus_one_; ++i) {
    positions_[offset + i] = position[i];
  }
  if (!cuda_enabled_) {
    return;
  }
  cudaMemcpy(d_positions_ + offset, position, dim_plus_one_ * sizeof(double), cudaMemcpyHostToDevice);
}

double LogLikelihoodWorkspace::compute_loglikelihood_sum(int v1,
                                                         const double *pos1,
                                                         const int *neighbors,
                                                         int neighbor_count,
                                                         double mu,
                                                         double beta,
                                                         double radius) {
  if (!cuda_enabled_) {
    return compute_loglikelihood_sum_cpu(dim_,
                                         nb_vertices_,
                                         positions_.data(),
                                         kappas_.data(),
                                         v1,
                                         pos1,
                                         neighbors,
                                         neighbor_count,
                                         mu,
                                         beta,
                                         radius);
  }

  cudaMemcpy(d_pos1_, pos1, dim_plus_one_ * sizeof(double), cudaMemcpyHostToDevice);
  if (neighbor_count > 0) {
    cudaMemcpy(d_neighbors_, neighbors, neighbor_count * sizeof(int), cudaMemcpyHostToDevice);
  }

  const int threads = 256;
  const int blocks_all = (nb_vertices_ + threads - 1) / threads;
  loglikelihood_all_kernel<<<blocks_all, threads>>>(d_positions_,
                                                    d_kappas_,
                                                    nb_vertices_,
                                                    dim_plus_one_,
                                                    dim_,
                                                    mu,
                                                    beta,
                                                    radius,
                                                    d_pos1_,
                                                    v1,
                                                    kappas_[v1],
                                                    d_output_all_);
  cudaDeviceSynchronize();

  const int blocks_neighbors = (neighbor_count + threads - 1) / threads;
  if (neighbor_count > 0) {
    loglikelihood_neighbors_kernel<<<blocks_neighbors, threads>>>(d_positions_,
                                                                  d_kappas_,
                                                                  d_neighbors_,
                                                                  neighbor_count,
                                                                  dim_plus_one_,
                                                                  dim_,
                                                                  mu,
                                                                  beta,
                                                                  radius,
                                                                  d_pos1_,
                                                                  kappas_[v1],
                                                                  d_output_neighbors_);
    cudaDeviceSynchronize();
  }

  const auto device_all = thrust::device_pointer_cast(d_output_all_);
  double sum_all = thrust::reduce(device_all, device_all + nb_vertices_, 0.0, thrust::plus<double>());
  double sum_neighbors = 0.0;
  if (neighbor_count > 0) {
    const auto device_neighbors = thrust::device_pointer_cast(d_output_neighbors_);
    sum_neighbors = thrust::reduce(device_neighbors, device_neighbors + neighbor_count, 0.0, thrust::plus<double>());
  }
  return sum_all + sum_neighbors;
}

bool LogLikelihoodWorkspace::cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

LogLikelihoodWorkspace1D::LogLikelihoodWorkspace1D(int nb_vertices)
    : nb_vertices_(nb_vertices) {
  thetas_.resize(nb_vertices_);
  kappas_.resize(nb_vertices_);

  if (!cuda_available()) {
    cuda_enabled_ = false;
    return;
  }

  const std::size_t thetas_bytes = static_cast<std::size_t>(nb_vertices_) * sizeof(double);
  const std::size_t kappas_bytes = static_cast<std::size_t>(nb_vertices_) * sizeof(double);
  const std::size_t output_bytes = static_cast<std::size_t>(nb_vertices_) * sizeof(double);
  const std::size_t neighbors_bytes = static_cast<std::size_t>(nb_vertices_) * sizeof(int);

  cuda_enabled_ = cuda_call_succeeded(cudaMalloc(&d_thetas_, thetas_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_kappas_, kappas_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_output_all_, output_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_output_neighbors_, output_bytes)) &&
                  cuda_call_succeeded(cudaMalloc(&d_neighbors_, neighbors_bytes));

  if (!cuda_enabled_) {
    if (d_thetas_) {
      cudaFree(d_thetas_);
    }
    if (d_kappas_) {
      cudaFree(d_kappas_);
    }
    if (d_output_all_) {
      cudaFree(d_output_all_);
    }
    if (d_output_neighbors_) {
      cudaFree(d_output_neighbors_);
    }
    if (d_neighbors_) {
      cudaFree(d_neighbors_);
    }
    d_thetas_ = nullptr;
    d_kappas_ = nullptr;
    d_output_all_ = nullptr;
    d_output_neighbors_ = nullptr;
    d_neighbors_ = nullptr;
  }
}

LogLikelihoodWorkspace1D::~LogLikelihoodWorkspace1D() {
  if (d_thetas_) {
    cudaFree(d_thetas_);
  }
  if (d_kappas_) {
    cudaFree(d_kappas_);
  }
  if (d_output_all_) {
    cudaFree(d_output_all_);
  }
  if (d_output_neighbors_) {
    cudaFree(d_output_neighbors_);
  }
  if (d_neighbors_) {
    cudaFree(d_neighbors_);
  }
}

bool LogLikelihoodWorkspace1D::is_cuda_enabled() const {
  return cuda_enabled_;
}

void LogLikelihoodWorkspace1D::set_thetas(const double *thetas, std::size_t count) {
  thetas_.assign(thetas, thetas + count);
  if (!cuda_enabled_) {
    return;
  }
  cudaMemcpy(d_thetas_, thetas_.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void LogLikelihoodWorkspace1D::set_kappas(const double *kappas, std::size_t count) {
  kappas_.assign(kappas, kappas + count);
  if (!cuda_enabled_) {
    return;
  }
  cudaMemcpy(d_kappas_, kappas_.data(), count * sizeof(double), cudaMemcpyHostToDevice);
}

void LogLikelihoodWorkspace1D::update_theta(int index, double theta) {
  thetas_[static_cast<std::size_t>(index)] = theta;
  if (!cuda_enabled_) {
    return;
  }
  cudaMemcpy(d_thetas_ + index, &theta, sizeof(double), cudaMemcpyHostToDevice);
}

double LogLikelihoodWorkspace1D::compute_loglikelihood_sum(int v1,
                                                           double theta1,
                                                           const int *neighbors,
                                                           int neighbor_count,
                                                           double mu,
                                                           double beta) {
  if (!cuda_enabled_) {
    return compute_loglikelihood_sum_cpu_1d(nb_vertices_,
                                            thetas_.data(),
                                            kappas_.data(),
                                            v1,
                                            theta1,
                                            neighbors,
                                            neighbor_count,
                                            mu,
                                            beta);
  }

  if (neighbor_count > 0) {
    cudaMemcpy(d_neighbors_, neighbors, neighbor_count * sizeof(int), cudaMemcpyHostToDevice);
  }

  const int threads = 256;
  const int blocks_all = (nb_vertices_ + threads - 1) / threads;
  loglikelihood_all_kernel_1d<<<blocks_all, threads>>>(d_thetas_,
                                                      d_kappas_,
                                                      nb_vertices_,
                                                      mu,
                                                      beta,
                                                      theta1,
                                                      v1,
                                                      kappas_[v1],
                                                      d_output_all_);
  cudaDeviceSynchronize();

  const int blocks_neighbors = (neighbor_count + threads - 1) / threads;
  if (neighbor_count > 0) {
    loglikelihood_neighbors_kernel_1d<<<blocks_neighbors, threads>>>(d_thetas_,
                                                                    d_kappas_,
                                                                    d_neighbors_,
                                                                    neighbor_count,
                                                                    nb_vertices_,
                                                                    mu,
                                                                    beta,
                                                                    theta1,
                                                                    kappas_[v1],
                                                                    d_output_neighbors_);
    cudaDeviceSynchronize();
  }

  const auto device_all = thrust::device_pointer_cast(d_output_all_);
  double sum_all = thrust::reduce(device_all, device_all + nb_vertices_, 0.0, thrust::plus<double>());
  double sum_neighbors = 0.0;
  if (neighbor_count > 0) {
    const auto device_neighbors = thrust::device_pointer_cast(d_output_neighbors_);
    sum_neighbors = thrust::reduce(device_neighbors, device_neighbors + neighbor_count, 0.0, thrust::plus<double>());
  }
  return sum_all + sum_neighbors;
}

bool LogLikelihoodWorkspace1D::cuda_available() {
  return LogLikelihoodWorkspace::cuda_available();
}

} // namespace embeddingSD_cuda
