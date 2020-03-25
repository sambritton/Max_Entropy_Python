
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void dispatch(const std::vector<int>& indices, const std::vector<double>& variables) {
    std::cout << "Dispatching jobs...\nGot indices:\n";

    for (const auto& i : indices)
    {
        std::cout << i << ", ";
    }
    std::cout << "\nGot variables:\n";

    for (const auto& i : variables)
    {
        std::cout << i << ", ";
    }
    std::cout << "\n";
}

PYBIND11_MODULE(pstep, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("dispatch", &dispatch, "A function which adds two numbers");
}
