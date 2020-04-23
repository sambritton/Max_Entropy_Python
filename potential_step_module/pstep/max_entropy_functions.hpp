
#ifndef __MEP_HPP
#define __MEP_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

[[nodiscard]]
constexpr auto calc_deltaS_metab(
        const double& v_log_counts,
        const double& target_v_log_counts
        ) -> double
{
    return v_log_counts - target_v_log_counts;
}

[[nodiscard]]
auto calc_reg_E_step(
        const std::vector<double>& E_vec,
        const std::vector<double>& React_Choice,
        const std::vector<double>& nvar,
        const std::vector<double>& log_vcounts,
        const std::vector<double>& log_fcounts,
        const std::vector<double>& complete_target_log_counts,
        const std::vector<double>& S_mat,
        const std::vector<double>& A,
        const std::vector<double>& rxn_flux,
        const std::vector<double>& KQ,
        const std::vector<double>& delta_S_metab
        ) -> double
{

    const int           nargin = delta_S_metab.size();
    constexpr int       method = 1;
    constexpr double    delta_S_val_method1 = 0.0;
    std::vector<double> vcounts = log_vcounts;
    std::vector<double> fcounts = log_fcounts;

    for (int i = 0; i < vcounts.size(); i++)
    {
        vcounts[i] = exp(vcounts[i]);
    }

    for (int i = 0; i < vcounts.size(); i++)
    {
        fcounts[i] = exp(fcounts[i]);
    }

    return 0.0;
}
#endif
