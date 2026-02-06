#ifndef DMERCATOR_CUDA_HPP_INCLUDED
#define DMERCATOR_CUDA_HPP_INCLUDED

#include <string>
#include <vector>

namespace dmercator
{
namespace cuda
{

// GPU backend for the hot likelihood-scoring paths used during refinement.
// The rest of the embedding code remains CPU-side and calls this through a thin API.
class LikelihoodBackend
{
  public:
    LikelihoodBackend();
    ~LikelihoodBackend();

    LikelihoodBackend(const LikelihoodBackend &) = delete;
    LikelihoodBackend &operator=(const LikelihoodBackend &) = delete;
    LikelihoodBackend(LikelihoodBackend &&) = delete;
    LikelihoodBackend &operator=(LikelihoodBackend &&) = delete;

    bool initialize(int nb_vertices,
                    const std::vector<int> &row_offsets,
                    const std::vector<int> &col_indices,
                    std::string *error_message);

    bool is_initialized() const;
    int nb_vertices() const;

    bool update_kappa(const std::vector<double> &kappa, std::string *error_message);
    bool download_kappa(std::vector<double> &kappa, std::string *error_message);
    bool update_observed_degree(const std::vector<int> &degree, std::string *error_message);
    bool update_theta(const std::vector<double> &theta, std::string *error_message);
    bool update_theta_entry(int vertex_id, double theta_value, std::string *error_message);

    // Positions are stored in SoA layout: [coord0(all vertices), coord1(all vertices), ...].
    bool update_positions_soa(int dim_plus_one,
                              const std::vector<double> &positions_soa,
                              std::string *error_message);
    bool update_position_entry(int vertex_id,
                               const std::vector<double> &position,
                               std::string *error_message);

    // Scores candidates for the S^1 local objective used by refine_angle(v1).
    bool score_candidates_s1(int v1,
                             double prefactor,
                             double beta,
                             const std::vector<double> &candidate_theta,
                             std::vector<double> &out_scores,
                             std::string *error_message);

    // Scores candidates for the S^D local objective used by refine_angle(dim, v1, radius).
    // candidate_positions_soa is [coord0(all candidates), coord1(all candidates), ...].
    bool score_candidates_sd(int dim,
                             int v1,
                             double radius,
                             double mu,
                             double beta,
                             const std::vector<double> &candidate_positions_soa,
                             std::vector<double> &out_scores,
                             std::string *error_message);

    // Executes one CUDA iteration of hidden-degree refinement in S^1.
    // Returns the maximal per-vertex expected-degree mismatch for convergence checks.
    bool run_kappa_iteration_s1(double prefactor,
                                double beta,
                                double convergence_threshold,
                                unsigned long long seed,
                                int iteration,
                                double *out_max_abs_error,
                                std::string *error_message);

    // Executes one CUDA iteration of hidden-degree refinement in S^D.
    bool run_kappa_iteration_sd(int dim,
                                double radius,
                                double mu,
                                double beta,
                                double convergence_threshold,
                                unsigned long long seed,
                                int iteration,
                                double *out_max_abs_error,
                                std::string *error_message);

    // Monte Carlo estimate of the expected mean clustering in S^1.
    bool estimate_mean_clustering_s1(double prefactor,
                                     double beta,
                                     unsigned long long seed,
                                     int nb_samples,
                                     double *out_mean_clustering,
                                     std::string *error_message);

    // Monte Carlo estimate of the expected mean clustering in S^D.
    bool estimate_mean_clustering_sd(int dim,
                                     double radius,
                                     double mu,
                                     double beta,
                                     unsigned long long seed,
                                     int nb_samples,
                                     double *out_mean_clustering,
                                     std::string *error_message);

    // Samples one synthetic graph in S^1 and returns upper-triangular edge flags (i<j).
    bool sample_graph_s1(double prefactor,
                         double beta,
                         unsigned long long seed,
                         std::vector<unsigned char> &out_upper_triangle_edges,
                         std::string *error_message);

    // Samples one synthetic graph in S^D and returns upper-triangular edge flags (i<j).
    bool sample_graph_sd(int dim,
                         double radius,
                         double mu,
                         double beta,
                         unsigned long long seed,
                         std::vector<unsigned char> &out_upper_triangle_edges,
                         std::string *error_message);

    // Computes histogram accumulators used by *.inf_pconn in S^1.
    bool histogram_pconn_s1(double prefactor,
                            double beta,
                            const std::vector<double> &bin_upper_bounds,
                            std::vector<double> &out_n,
                            std::vector<double> &out_p,
                            std::vector<double> &out_x,
                            std::string *error_message);

    // Computes histogram accumulators used by *.inf_pconn in S^D.
    bool histogram_pconn_sd(int dim,
                            double radius,
                            double mu,
                            double beta,
                            const std::vector<double> &bin_upper_bounds,
                            std::vector<double> &out_n,
                            std::vector<double> &out_p,
                            std::vector<double> &out_x,
                            std::string *error_message);

  private:
    struct Impl;
    Impl *impl_;
};

} // namespace cuda
} // namespace dmercator

#endif
