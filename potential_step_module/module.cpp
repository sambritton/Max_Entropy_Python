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
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include "helper_functions.hpp"

namespace py = pybind11;

/*
 *  %%
 *  input must be able to determine optimization routine. We need the follwing variables:
 *  0: state, type:np.array(float), purpose: current enzyme activities
 *  1: v_log_counts, type:np.array(float), purpose:initial guess for optimization
 *  2: f_log_counts, type:np.array(float), purpose:fixed metabolites
 *  3: mu0, type: , purpose: non as of now
 *  4: S_mat, type: np.array(float), purpose: stoichiometric matrix (rxn by matabolites)
 *  5: R_back_mat, type: np.array(float), purpose: reverse stoichiometric matrix (rxn by matabolites)
 *         #note could be calculated from S_mat: R_back_mat = np.where(S_mat<0, S_mat, 0)
 *  6: P_mat, type: np.array(float), purpose: forward stoichiometric matrix (rxn by matabolites), 
 *         # note could be calculated from S_mat: P_mat = np.where(S_mat>0,S_mat,0)
 *  7: Keq_constant, type: np.array(float), purpose: equilibrium constants
 *
 */
void potential_step(
        const int index,
        const Eigen::VectorXd& S_mat,
        const Eigen::VectorXd& R_back_mat,
        const Eigen::VectorXd& P_mat,
        const Eigen::VectorXd& Keq_constant,
        const Eigen::VectorXd& E_Regulation,
        const Eigen::VectorXd& log_fcounts,
        double* returns,
        int tid)
{
    std::cout << "--- tid<" << tid << "> potential step being calculated...\n";

    Eigen::VectorXd result = least_squares(
            S_mat,
            R_back_mat,
            P_mat, 
            Keq_constant,
            E_Regulation,
            log_fcounts);

    returns[tid] = result(0);
    std::cout << "Returning from tid<" << tid << ">\n";
}


[[nodiscard]] auto dispatch(
        const std::vector<int>& indices,
        std::vector<Eigen::VectorXd>& variables
        ) -> Eigen::VectorXd
{
    if constexpr (MY_CPP_STD < CPP11)
    {
        Eigen::initParallel();
    }

    // No need for parallelism within Eigen, as each thread will be
    // using the Eigen library on it's own thread -
    // unless of course we run in a slurm allocation and we have
    // enough threads both for each call of potential_step and for
    // the parallelism within Eigen.
    //
    // Eigen::setNbThreads(n_threads);

    assert(variables.size() == 6 && "Did you pass in all 8 variables as numpy arrays?");
    const Eigen::VectorXd& S_mat 		= variables[0];
    const Eigen::VectorXd& R_back_mat 	= variables[1];
    const Eigen::VectorXd& P_mat 		= variables[2];
    const Eigen::VectorXd& Keq_constant = variables[3];
    const Eigen::VectorXd& E_Regulation = variables[4];
    const Eigen::VectorXd& log_fcounts  = variables[5];
    const int n_threads = indices.size();
    double* returns = new double[n_threads];

    for (int tid=0; tid<n_threads; tid++)
    {
        potential_step(
                indices[tid],
                S_mat,
                R_back_mat,
                P_mat,
                Keq_constant,
                E_Regulation,
                log_fcounts,
                returns,
                tid);
    }

    /*
     * Not used for now... Sequential solver first!
     *
    std::vector<std::thread> handles(n_threads);
    for (int tid=0; tid<n_threads; tid++)
    {
        handles[tid] = std::thread(
                potential_step,
                indices[tid],
                std::ref(state),
                std::ref(v_log_counts),
                std::ref(f_log_counts),
                std::ref(mu0),
                std::ref(S_mat),
                std::ref(R_back_mat),
                std::ref(P_mat),
                std::ref(Keq_constant),
                returns,
                tid
            );
    }

    for (auto& th : handles) th.join();
    */

    std::cout << "Graceful exit...\n";
}

PYBIND11_MODULE(pstep, m) {
    m.doc() = "Dispatches jobs to calculate potential steps."; // optional module docstring
    m.def("dispatch", &dispatch, "Dispatches jobs");
}
