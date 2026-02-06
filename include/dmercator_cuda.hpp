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

  private:
    struct Impl;
    Impl *impl_;
};

} // namespace cuda
} // namespace dmercator

#endif
