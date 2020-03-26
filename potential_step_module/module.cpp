
#include <vector>
#include <iostream>
#include <typeinfo>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void dispatch(const std::vector<int>& indices, py::dict variables) {
    auto nn_model		                    = variables["nn_model"];
    auto state		                        = variables["state"];
    auto nvar		                        = variables["nvar"];
    auto v_log_counts		                = variables["v_log_counts"];
    auto f_log_counts 		                = variables["f_log_counts"];
    auto complete_target_log_counts		    = variables["complete_target_log_counts"];
    auto A		                            = variables["A"];
    auto rxn_flux		                    = variables["rxn_flux"];
    auto KQ_f 		                        = variables["KQ_f"];
    auto delta_S_metab 		                = variables["delta_S_metab"];
    auto mu0		                        = variables["mu0"];
    auto S_mat		                        = variables["S_mat"];
    auto R_back_mat		                    = variables["R_back_mat"];
    auto P_mat 		                        = variables["P_mat"];
    auto delta_increment_for_small_concs	= variables["delta_increment_for_small_concs"];
    auto Keq_constant		                = variables["Keq_constant"];

    for (const auto& el : variables)
    {
        std::cout << "Got key " << el.first << " with type " << typeid(el.first).name() << "\n";
        std::cout << "and value " << el.second << " with type " << typeid(el.second).name() << "\n\n";
    }
}

PYBIND11_MODULE(pstep, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("dispatch", &dispatch, "A function which adds two numbers");
}
