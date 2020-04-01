
#ifndef __HELPER_FUNCTIONS_H
#define __HELPER_FUNCTIONS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  const int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }
};

struct LMFunctor : Functor<double>
{
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
    LMFunctor(void): Functor<double>(3,15) {}

    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double tmp1, tmp2, tmp3;
        static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }

    int df(const VectorXd &x, Eigen::MatrixXd &fjac) const
    {
        double tmp1, tmp2, tmp3, tmp4;
        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4 = tmp4*tmp4;
            fjac(i,0) = -1;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
        return 0;
    }
};

[[nodiscard]]
constexpr auto odds_alternate(
            const Eigen::VectorXd& E_regulation,
            const Eigen::VectorXd& log_metabolites,
            const Eigen::VectorXd& mu0,
            const Eigen::VectorXd& S_mat,
            const Eigen::VectorXd& R_back_mat,
            const Eigen::VectorXd& P_mat, 
            const Eigen::VectorXd& Keq_constant,
            const int direction=1) -> double
{
    Eigen::VectorXd log_Q_inv = -direction * (R_back_mat.dot(log_metabolites) + P_mat.dot(log_metabolites));
    const double scale_min  = log_Q_inv.min();
    const double scale_max  = log_Q_inv.max();
    const double scale      = (scale_max + scale_min) / 2.0;

    const Eigen::VectorXd scaled_val = log_Q_inv - scale;
    const Eigen::VectorXd log_EKQ = (E_regulation * Keq_constant).log() + log_Q_inv;
    const double q_max = log_Q_inv.abs().max();
    const double ekq_max = log_EKQ.abs().max();

    Eigen::VectorXd EKQ;
    if (q_max < ekq_max)
    {
        EKQ = E_Regulation * (log_Q_inv.exp() * Keq_constant);
    }
    else
    {
        EKQ = ((E_Regulation * Keq_constant).log() + log_Q_inv).exp()
    }

    return EKQ;
}

[[nodiscard]]
double derivatives(
            const Eigen::VectorXd& v_log_counts,
            const Eigen::VectorXd& f_log_counts,
            const Eigen::VectorXd& mu0,
            const Eigen::VectorXd& S_mat,
            const Eigen::VectorXd& R_back_mat,
            const Eigen::VectorXd& P_mat, 
            const Eigen::VectorXd& Keq_constant,
            const Eigen::VectorXd& E_regulation)
{
    const int n_var = v_log_counts.size();
    const Eigen::VectorXd log_metabolites(v_log_counts.size() + f_log_counts.size());
    log_metabolites << v_log_counts, f_log_counts;

    EKQ_f = odds_alternate(
            E_regulation,
            log_metabolites,
            mu0,
            S_mat,
            R_back_mat,
            P_mat,
            Keq_constant,
            1);

    Eigen::VectorXd EKQ_r = odds_alternate(
            E_regulation,
            log_metabolites,
            mu0,
            -S_mat,
            P_mat,
            R_back_mat,
            Keq_constant.inverse(),
            1);

    Eigen::MatrixXd s_mat = S_mat(all, seqN(0, nvar));
    Eigen::MatrixXd deriv = s_mat.transpose().dot((EKQ_f - EKQ_r).transpose());
    return deriv.reshaped(deriv.size(), 1);
}

/*
 * Driver function for calculating the least squares with the
 * Levenberg-Marquardt method
 */
[[nodiscard]]
Eigen::Vector2d least_squares(
            const Eigen::VectorXd& v_log_counts,
            const double           xtol,
            const Eigen::VectorXd& f_log_counts,
            const Eigen::VectorXd& mu0,
            const Eigen::VectorXd& S_mat,
            const Eigen::VectorXd& R_back_mat,
            const Eigen::VectorXd& P_mat, 
            const Eigen::VectorXd& Keq_constant,
            const Eigen::VectorXd& state)
{
    Eigen::Vector2d x;
    static constexpr int n = 3;
    x.setConstant(n, 1.);
    LMFunctor lmfn;
    LevenbergMarquardt<LMFunctor> lm(lmfn);
    lm.minimize(x);

    return x;
}

[[nodiscard]] static inline
auto calc_new_enzyme_simple(
        const Eigen::VectorXd& state,
        const int& index) -> double
{
    double current_e = state[index];
    return current_e - ( current_e / 5.0 );
}

#endif
