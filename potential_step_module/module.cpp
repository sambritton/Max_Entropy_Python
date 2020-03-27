
#include <vector>
#include <iostream>
#include <typeinfo>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

constexpr int potential_step()
{
    return 0;
}

void dispatch(const std::vector<int>& indices, py::dict variables) {
    const auto nn_model		                    = variables["nn_model"];
    const auto state		                    = variables["state"];
    const auto nvar		                        = variables["nvar"];
    const auto v_log_counts		                = variables["v_log_counts"];
    const auto f_log_counts 		            = variables["f_log_counts"];
    const auto complete_target_log_counts		= variables["complete_target_log_counts"];
    const auto A		                        = variables["A"];
    const auto rxn_flux		                    = variables["rxn_flux"];
    const auto KQ_f 		                    = variables["KQ_f"];
    const auto delta_S_metab 		            = variables["delta_S_metab"];
    const auto mu0		                        = variables["mu0"];
    const auto S_mat		                    = variables["S_mat"];
    const auto R_back_mat		                = variables["R_back_mat"];
    const auto P_mat 		                    = variables["P_mat"];
    const auto delta_increment_for_small_concs	= variables["delta_increment_for_small_concs"];
    const auto Keq_constant		                = variables["Keq_constant"];

    for (const auto& el : variables)
    {
        std::cout << "Got key " << el.first << " with type " << typeid(el.first).name() << "\n";
        std::cout << "and value " << el.second << " with type " << typeid(el.second).name() << "\n\n";
    }
}

PYBIND11_MODULE(pstep, m) {
    m.doc() = "Dispatches jobs to calculate potential steps."; // optional module docstring
    m.def("dispatch", &dispatch, "Dispatches jobs");
}
