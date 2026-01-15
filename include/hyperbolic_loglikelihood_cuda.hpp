#ifndef HYPERBOLIC_LOGLIKELIHOOD_CUDA_HPP_INCLUDED
#define HYPERBOLIC_LOGLIKELIHOOD_CUDA_HPP_INCLUDED

#include <cmath>
#include <cstddef>
#include <vector>

namespace embeddingSD_cuda {

constexpr double kNumericalZero = 1e-10;
constexpr double kPi = 3.14159265358979323846;

inline double compute_angle_d_vectors_flat(const double *v1,
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
  norm1 /= std::sqrt(norm1);
  norm2 /= std::sqrt(norm2);
  const double result = dot / (norm1 * norm2);
  if (std::fabs(result - 1.0) < kNumericalZero) {
    return 0.0;
  }
  return std::acos(result);
}

inline double compute_loglikelihood_sum_cpu(int dim,
                                            int nb_vertices,
                                            const double *positions,
                                            const double *kappas,
                                            int v1,
                                            const double *pos1,
                                            const int *neighbors,
                                            int neighbor_count,
                                            double mu,
                                            double beta,
                                            double radius) {
  const int dim_plus_one = dim + 1;
  const double kappa1 = kappas[v1];
  double sum = 0.0;

  for (int v2 = 0; v2 < nb_vertices; ++v2) {
    if (v1 == v2) {
      continue;
    }
    const double *pos2 = positions + (v2 * dim_plus_one);
    const double dtheta = compute_angle_d_vectors_flat(pos1, pos2, dim_plus_one);
    const double chi = radius * dtheta / std::pow(mu * kappa1 * kappas[v2], 1.0 / dim);
    const double prob = 1.0 / (1.0 + std::pow(chi, beta));
    sum += std::log(1.0 - prob);
  }

  for (int i = 0; i < neighbor_count; ++i) {
    const int v2 = neighbors[i];
    if (v1 == v2) {
      continue;
    }
    const double *pos2 = positions + (v2 * dim_plus_one);
    const double dtheta = compute_angle_d_vectors_flat(pos1, pos2, dim_plus_one);
    const double chi = radius * dtheta / std::pow(mu * kappa1 * kappas[v2], 1.0 / dim);
    const double prob = 1.0 / (1.0 + std::pow(chi, beta));
    sum += std::log(prob);
  }

  return sum;
}

inline double compute_loglikelihood_sum_cpu_1d(int nb_vertices,
                                               const double *thetas,
                                               const double *kappas,
                                               int v1,
                                               double theta1,
                                               const int *neighbors,
                                               int neighbor_count,
                                               double mu,
                                               double beta) {
  double sum = 0.0;
  const double kappa1 = kappas[v1];
  for (int v2 = 0; v2 < nb_vertices; ++v2) {
    if (v1 == v2) {
      continue;
    }
    const double da = kPi - std::fabs(kPi - std::fabs(theta1 - thetas[v2]));
    const double fraction = (nb_vertices * da) / (2.0 * kPi * mu * kappa1 * kappas[v2]);
    sum += -std::log(1.0 + std::pow(fraction, -beta));
  }
  for (int i = 0; i < neighbor_count; ++i) {
    const int v2 = neighbors[i];
    if (v1 == v2) {
      continue;
    }
    const double da = kPi - std::fabs(kPi - std::fabs(theta1 - thetas[v2]));
    const double fraction = (nb_vertices * da) / (2.0 * kPi * mu * kappa1 * kappas[v2]);
    sum += -beta * std::log(fraction);
  }
  return sum;
}

#ifdef MERCATOR_USE_CUDA
class LogLikelihoodWorkspace {
  public:
    LogLikelihoodWorkspace(int dim, int nb_vertices);
    ~LogLikelihoodWorkspace();
    LogLikelihoodWorkspace(const LogLikelihoodWorkspace&) = delete;
    LogLikelihoodWorkspace& operator=(const LogLikelihoodWorkspace&) = delete;

    bool is_cuda_enabled() const;
    void set_positions(const double *positions, std::size_t count);
    void set_kappas(const double *kappas, std::size_t count);
    void update_position(int index, const double *position);
    double compute_loglikelihood_sum(int v1,
                                     const double *pos1,
                                     const int *neighbors,
                                     int neighbor_count,
                                     double mu,
                                     double beta,
                                     double radius);
    static bool cuda_available();

  private:
    int dim_;
    int nb_vertices_;
    int dim_plus_one_;
    bool cuda_enabled_{false};
    std::vector<double> positions_;
    std::vector<double> kappas_;
    double *d_positions_{nullptr};
    double *d_kappas_{nullptr};
    double *d_pos1_{nullptr};
    double *d_output_all_{nullptr};
    double *d_output_neighbors_{nullptr};
    int *d_neighbors_{nullptr};
};

class LogLikelihoodWorkspace1D {
  public:
    LogLikelihoodWorkspace1D(int nb_vertices);
    ~LogLikelihoodWorkspace1D();
    LogLikelihoodWorkspace1D(const LogLikelihoodWorkspace1D&) = delete;
    LogLikelihoodWorkspace1D& operator=(const LogLikelihoodWorkspace1D&) = delete;

    bool is_cuda_enabled() const;
    void set_thetas(const double *thetas, std::size_t count);
    void set_kappas(const double *kappas, std::size_t count);
    void update_theta(int index, double theta);
    double compute_loglikelihood_sum(int v1,
                                     double theta1,
                                     const int *neighbors,
                                     int neighbor_count,
                                     double mu,
                                     double beta);
    static bool cuda_available();

  private:
    int nb_vertices_;
    bool cuda_enabled_{false};
    std::vector<double> thetas_;
    std::vector<double> kappas_;
    double *d_thetas_{nullptr};
    double *d_kappas_{nullptr};
    double *d_output_all_{nullptr};
    double *d_output_neighbors_{nullptr};
    int *d_neighbors_{nullptr};
};
#else
class LogLikelihoodWorkspace {
  public:
    LogLikelihoodWorkspace(int dim, int nb_vertices)
        : dim_(dim), nb_vertices_(nb_vertices), dim_plus_one_(dim + 1) {}
    ~LogLikelihoodWorkspace() = default;
    LogLikelihoodWorkspace(const LogLikelihoodWorkspace&) = delete;
    LogLikelihoodWorkspace& operator=(const LogLikelihoodWorkspace&) = delete;

    bool is_cuda_enabled() const { return false; }
    void set_positions(const double *positions, std::size_t count) {
      positions_.assign(positions, positions + count);
    }
    void set_kappas(const double *kappas, std::size_t count) {
      kappas_.assign(kappas, kappas + count);
    }
    void update_position(int index, const double *position) {
      const std::size_t offset = static_cast<std::size_t>(index) * dim_plus_one_;
      for (int i = 0; i < dim_plus_one_; ++i) {
        positions_[offset + i] = position[i];
      }
    }
    double compute_loglikelihood_sum(int v1,
                                     const double *pos1,
                                     const int *neighbors,
                                     int neighbor_count,
                                     double mu,
                                     double beta,
                                     double radius) {
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
    static bool cuda_available() { return false; }

  private:
    int dim_;
    int nb_vertices_;
    int dim_plus_one_;
    std::vector<double> positions_;
    std::vector<double> kappas_;
};

class LogLikelihoodWorkspace1D {
  public:
    LogLikelihoodWorkspace1D(int nb_vertices) : nb_vertices_(nb_vertices) {}
    ~LogLikelihoodWorkspace1D() = default;
    LogLikelihoodWorkspace1D(const LogLikelihoodWorkspace1D&) = delete;
    LogLikelihoodWorkspace1D& operator=(const LogLikelihoodWorkspace1D&) = delete;

    bool is_cuda_enabled() const { return false; }
    void set_thetas(const double *thetas, std::size_t count) {
      thetas_.assign(thetas, thetas + count);
    }
    void set_kappas(const double *kappas, std::size_t count) {
      kappas_.assign(kappas, kappas + count);
    }
    void update_theta(int index, double theta) {
      thetas_[static_cast<std::size_t>(index)] = theta;
    }
    double compute_loglikelihood_sum(int v1,
                                     double theta1,
                                     const int *neighbors,
                                     int neighbor_count,
                                     double mu,
                                     double beta) {
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
    static bool cuda_available() { return false; }

  private:
    int nb_vertices_;
    std::vector<double> thetas_;
    std::vector<double> kappas_;
};
#endif

} // namespace embeddingSD_cuda

#endif
