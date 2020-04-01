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
        Eigen::VectorXd& state,
        const Eigen::VectorXd& v_log_counts,
        const Eigen::VectorXd& f_log_counts,
        const Eigen::VectorXd& mu0,
        const Eigen::VectorXd& S_mat,
        const Eigen::VectorXd& R_back_mat,
        const Eigen::VectorXd& P_mat,
        const Eigen::VectorXd& Keq_constant,
        double* returns,
        int tid)
{
    (void) index;
    (void) state;
    (void) v_log_counts;
    (void) f_log_counts;
    (void) mu0;
    (void) S_mat;
    (void) R_back_mat;
    (void) P_mat;
    (void) Keq_constant;
    std::cout << "--- tid<" << tid << "> potential step being calculated...\n";
    P_mat << "This should fail!!";
    /*
     *  This function shoud run for each reaction (index = 0 : nrxn-1)
     *  It will apply regulation to the state (enzyme activities) 
     *  and calculate resulting steady state metabolite concentrations
     */

    state[index] = calc_new_enzyme_simple(state, index);

    Eigen::Vector2d result = least_squares(
            v_log_counts,
            1e-15,
            f_log_counts,
            mu0,
            S_mat,
            R_back_mat,
            P_mat, 
            Keq_constant,
            state);
    /*
     *  def potential_step(index, other_args):
     *      React_Choice=index
     *      
     *      state, v_log_counts, f_log_counts,\
     *      mu0, S_mat, R_back_mat, P_mat, \
     *      delta_increment_for_small_concs, Keq_constant = other_args
     *      
     *      newE = max_entropy_functions.calc_new_enzyme_simple(state, React_Choice)
     *      trial_state_sample = state.copy()#DO NOT MODIFY ORIGINAL STATE
     *      trial_state_sample[React_Choice] = newE
     *      new_res_lsq = least_squares(
     *              max_entropy_functions.derivatives, v_log_counts, method='lm',
     *              xtol=1e-15, 
     *              args=(
     *                  f_log_counts,
     *                  mu0,
     *                  S_mat,
     *                  R_back_mat,
     *                  P_mat, 
     *                  delta_increment_for_small_concs,
     *                  Keq_constant,
     *                  trial_state_sample
     *              )
     *          )
     */
    returns[tid] = result(0);
    std::cout << "Returning from tid<" << tid << ">\n";
}


void dispatch(const std::vector<int>& indices, std::vector<Eigen::VectorXd>& variables)
{
    assert(variables.size() == 8 && "Did you pass in all 8 variables as numpy arrays?");
    Eigen::VectorXd& state 		= variables[0];
    const Eigen::VectorXd& v_log_counts = variables[1];
    const Eigen::VectorXd& f_log_counts = variables[2];
    const Eigen::VectorXd& mu0 			= variables[3];
    const Eigen::VectorXd& S_mat 		= variables[4];
    const Eigen::VectorXd& R_back_mat 	= variables[5];
    const Eigen::VectorXd& P_mat 		= variables[6];
    const Eigen::VectorXd& Keq_constant = variables[7];
    const int n_threads = indices.size();

    double* returns = (double*)malloc(sizeof(double) * n_threads);
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

    std::cout << "Graceful exit...\n";
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
