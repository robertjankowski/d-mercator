#include <cmath>
#include <iostream>
#include <vector>

#include "../include/hyperbolic_loglikelihood_cuda.hpp"

namespace {

bool nearly_equal(double a, double b, double tol = 1e-9) {
  return std::fabs(a - b) <= tol;
}

} // namespace

int main() {
  const int dim = 2;
  const int dim_plus_one = dim + 1;
  const int nb_vertices = 3;
  std::vector<double> positions = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0};
  std::vector<double> kappas = {1.0, 2.0, 3.0};
  const double mu = 0.5;
  const double beta = 1.7;
  const double radius = 1.0;

  const int v1 = 0;
  const double *pos1 = positions.data();
  std::vector<int> neighbors = {1};

  const double cpu_sum = embeddingSD_cuda::compute_loglikelihood_sum_cpu(
      dim,
      nb_vertices,
      positions.data(),
      kappas.data(),
      v1,
      pos1,
      neighbors.data(),
      static_cast<int>(neighbors.size()),
      mu,
      beta,
      radius);

  embeddingSD_cuda::LogLikelihoodWorkspace workspace(dim, nb_vertices);
  workspace.set_positions(positions.data(), positions.size());
  workspace.set_kappas(kappas.data(), kappas.size());

  const double workspace_sum = workspace.compute_loglikelihood_sum(
      v1,
      pos1,
      neighbors.data(),
      static_cast<int>(neighbors.size()),
      mu,
      beta,
      radius);

  if (!nearly_equal(cpu_sum, workspace_sum)) {
    std::cerr << "Mismatch in loglikelihood sum: cpu=" << cpu_sum
              << " workspace=" << workspace_sum << std::endl;
    return 1;
  }

  std::vector<double> updated = {0.0, 1.0, 1.0};
  const double updated_norm =
      std::sqrt(updated[0] * updated[0] + updated[1] * updated[1] + updated[2] * updated[2]);
  for (auto &val : updated) {
    val /= updated_norm;
  }
  workspace.update_position(1, updated.data());
  for (int i = 0; i < dim_plus_one; ++i) {
    positions[dim_plus_one + i] = updated[i];
  }

  const double cpu_sum_updated = embeddingSD_cuda::compute_loglikelihood_sum_cpu(
      dim,
      nb_vertices,
      positions.data(),
      kappas.data(),
      v1,
      pos1,
      neighbors.data(),
      static_cast<int>(neighbors.size()),
      mu,
      beta,
      radius);

  const double workspace_sum_updated = workspace.compute_loglikelihood_sum(
      v1,
      pos1,
      neighbors.data(),
      static_cast<int>(neighbors.size()),
      mu,
      beta,
      radius);

  if (!nearly_equal(cpu_sum_updated, workspace_sum_updated)) {
    std::cerr << "Mismatch after update: cpu=" << cpu_sum_updated
              << " workspace=" << workspace_sum_updated << std::endl;
    return 1;
  }

  {
    const int nb_vertices_1d = 4;
    std::vector<double> thetas = {0.1, 1.2, 2.0, 3.0};
    std::vector<double> kappas_1d = {1.0, 1.5, 2.0, 2.5};
    const double mu_1d = 0.4;
    const double beta_1d = 1.2;
    const int v1_1d = 2;
    const double theta1 = thetas[v1_1d];
    std::vector<int> neighbors_1d = {0, 3};

    const double cpu_sum_1d = embeddingSD_cuda::compute_loglikelihood_sum_cpu_1d(
        nb_vertices_1d,
        thetas.data(),
        kappas_1d.data(),
        v1_1d,
        theta1,
        neighbors_1d.data(),
        static_cast<int>(neighbors_1d.size()),
        mu_1d,
        beta_1d);

    embeddingSD_cuda::LogLikelihoodWorkspace1D workspace_1d(nb_vertices_1d);
    workspace_1d.set_thetas(thetas.data(), thetas.size());
    workspace_1d.set_kappas(kappas_1d.data(), kappas_1d.size());

    const double workspace_sum_1d = workspace_1d.compute_loglikelihood_sum(
        v1_1d,
        theta1,
        neighbors_1d.data(),
        static_cast<int>(neighbors_1d.size()),
        mu_1d,
        beta_1d);

    if (!nearly_equal(cpu_sum_1d, workspace_sum_1d)) {
      std::cerr << "Mismatch in 1D loglikelihood sum: cpu=" << cpu_sum_1d
                << " workspace=" << workspace_sum_1d << std::endl;
      return 1;
    }

    const double updated_theta = 2.4;
    workspace_1d.update_theta(1, updated_theta);
    thetas[1] = updated_theta;

    const double cpu_sum_1d_updated = embeddingSD_cuda::compute_loglikelihood_sum_cpu_1d(
        nb_vertices_1d,
        thetas.data(),
        kappas_1d.data(),
        v1_1d,
        theta1,
        neighbors_1d.data(),
        static_cast<int>(neighbors_1d.size()),
        mu_1d,
        beta_1d);

    const double workspace_sum_1d_updated = workspace_1d.compute_loglikelihood_sum(
        v1_1d,
        theta1,
        neighbors_1d.data(),
        static_cast<int>(neighbors_1d.size()),
        mu_1d,
        beta_1d);

    if (!nearly_equal(cpu_sum_1d_updated, workspace_sum_1d_updated)) {
      std::cerr << "Mismatch after 1D update: cpu=" << cpu_sum_1d_updated
                << " workspace=" << workspace_sum_1d_updated << std::endl;
      return 1;
    }
  }

  return 0;
}
