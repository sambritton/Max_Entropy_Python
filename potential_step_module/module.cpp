
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int dispatch(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(potential_step, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
}
