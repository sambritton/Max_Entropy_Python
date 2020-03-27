/*
 *
 * All we need to let someone else use this as a
 * package:
 *
 * Basic input S to calculate potential step
 * which metabolites are fixed and which are variable
 *
 */

#include <vector>
#include <iostream>
#include <typeinfo>
#include <thread>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "max_entropy_functions.hpp"

namespace py = pybind11;

constexpr int potential_step()
{
    /*
    newE = max_entropy_functions.calc_reg_E_step(
            state
            React_Choice
            nvar
            v_log_counts
            f_log_counts
            complete_target_log_counts
            S_mat
            A
            rxn_flux
            KQ_f
            delta_S_metab)
            */
    return 0;
}

/**
 * @inputs:
 * - current state of NN
 * - num_reactions (len of KQ)
 *
 */
void dispatch(const std::vector<int>& indices, py::dict variables) {

    /* Mutable * /
    auto nn_model		                = variables["nn_model"];
    auto state		                    = variables["state"];
    auto v_log_counts		            = variables["v_log_counts"];
    auto A		                        = variables["A"];
    auto rxn_flux		                = variables["rxn_flux"];
    auto delta_S_metab 		            = variables["delta_S_metab"];
    auto KQ_f 		                    = variables["KQ_f"];
    // -- */

    /* Unused... * /
    const auto mu0		                        = variables["mu0"];
    const auto delta_increment_for_small_concs	= variables["delta_increment_for_small_concs"];
    // -- */

    /* Constants: * /
    const auto f_log_counts 		            = variables["f_log_counts"];
    const auto nvar		                        = variables["nvar"];
    const auto complete_target_log_counts		= variables["complete_target_log_counts"];
    const auto Keq_constant		                = variables["Keq_constant"];

    const auto S_mat		                    = variables["S_mat"];
    const auto R_back_mat		                = variables["R_back_mat"];
    const auto P_mat 		                    = variables["P_mat"];
    // -- */

    for (const auto& el : variables)
    {
        std::cout << "Got key " << el.first << " with type " << typeid(el.first).name() << "\n";
        std::cout << "and value " << el.second << " with type " << typeid(el.second).name() << "\n\n";
    }

    /*
     * which level of granularity of parallelism
     * will best suit the problem?
     *
     */
}

PYBIND11_MODULE(pstep, m) {
    m.doc() = "Dispatches jobs to calculate potential steps."; // optional module docstring
    m.def("dispatch", &dispatch, "Dispatches jobs");
}
