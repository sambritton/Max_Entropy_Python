
#ifndef __HELPER_FUNCTIONS_H
#define __HELPER_FUNCTIONS_H

#include <cmath>
#include <Eigen/Eigen>
// #include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace Eigen;

static constexpr int CPP98      = 199711;
static constexpr int GCC98      = 199711;
static constexpr int CPP11      = 201103;
static constexpr int GNU11      = 201103;
static constexpr int CPP14      = 201402;
static constexpr int GNU14      = 201402;
static constexpr int CPP1z      = 201500;
static constexpr int CPP17      = 201500;
static constexpr int MY_CPP_STD = __cplusplus;
static constexpr int n_threads  = 8;

static constexpr double init_x = 0.001;

inline void printShape(const std::string& s, const Eigen::MatrixXd& m)
{
    std::cout << "Shape of " << s << ": (" << m.rows() << ", " << m.cols() << ")\n";
}

struct lmder_functor // : DenseFunctor<double>
{
    typedef double Scalar;
    typedef Eigen::Matrix<double, Dynamic, 1> InputType;
    typedef Eigen::Matrix<double, Dynamic, 1> ValueType;
    typedef Eigen::Matrix<double, Dynamic, Dynamic> JacobianType;

    MatrixXd S;
    MatrixXd R;
    MatrixXd P;
    VectorXd Keq_constant;
    VectorXd E_Regulation;
    VectorXd log_fcounts;

    // Number of data points, i.e. values.
    int m;

    // The number of parameters, i.e. inputs.
    int n;

    lmder_functor(
        MatrixXd& _S,
        MatrixXd& _R, 
        MatrixXd& _P,
        VectorXd& _Keq_constant,
        VectorXd& _E_Regulation,
        VectorXd& _log_fcounts):
        m(_S.rows()),
        n(_S.cols()-_log_fcounts.size()),
        S(_S),
        R(_R),
        P(_P),
        Keq_constant(_Keq_constant),
        E_Regulation(_E_Regulation),
        log_fcounts(_log_fcounts) {}
   
    // Returns 'm', the number of values.
    int values() const
    { 
        return m;
    }

    // Returns 'n', the number of inputs.
    int inputs() const
    { 
        return n;
    }
    
    int operator()(const VectorXd& log_vcounts, VectorXd& deriv)
    {
        std::cout << __func__ << " start\n";
        //this function should be derivs
    
        int nrxns = S.rows();
        int nvar = log_vcounts.rows();//make sure this is length and not 1
        int metabolite_count = S.cols();
        

        VectorXd log_metabolites(log_vcounts.size() + log_fcounts.size());
        log_metabolites << log_vcounts, log_fcounts;

        VectorXd log_Q_inv = -1.0 * ( (R * log_metabolites) + (P * log_metabolites));
        VectorXd log_Q = 1.0 * ( (P * log_metabolites) + (R * log_metabolites));
        std::cout << "log_Q_inv size: " << log_Q_inv.size()
            << "\tlog_Q size: " << log_Q.size() << "\n";

        VectorXd EKQ_f(nrxns);  //allocate. can break down to one vector but leave as two for clarity right now. 
        VectorXd EKQ_r(nrxns);    

        for (int rxn=0; rxn < nrxns; rxn++){
            double Q_inv_rxn = exp(log_Q_inv(rxn));
            double ekq_f = E_Regulation(rxn) * Keq_constant(rxn) * Q_inv_rxn;
            
            EKQ_f(rxn) = ekq_f;

            double Q_rxn = exp(log_Q(rxn));
            double ekq_r = E_Regulation(rxn) * pow(Keq_constant(rxn), -1.0) * Q_rxn; 
            EKQ_r(rxn) = ekq_r;
        }

        printShape("--- S", S);
        // auto _S = S.block(nrxns,nvar); //take all rows (reactions) and only variable columns.

        //(nvar x 1) <=(nvar x nrxns) * (nrxns x 1)
        deriv = (S.topLeftCorner(nrxns, nvar).transpose()) * (EKQ_f - EKQ_r);
        printShape("deriv", deriv);
        std::cout << __func__ << " end\n";
        return 0;
    }

    
    //WARNING  jacobian should be calculated wrt metabolite concentration, not log(concentration). 
    int df(const VectorXd &log_vcounts, MatrixXd &fjac)
    {
        std::cout << __func__ << " start\n";
        //this should be a numerical jacobian
        //Jac is the Jacobian matrix, 
        //an N metabolite time-differential equations by (rows) by 
        //N metabolites derivatives (columns)
        //J_ij = d/dx_i(df_j/dt)

        int nrxns = S.rows();
        int nvar = log_vcounts.rows();//make sure this is length and not 1
        int metabolite_count = S.cols();

        //WARNING Only use to calcualte KQ
        VectorXd log_metabolites(log_vcounts.size() + log_fcounts.size());
        log_metabolites << log_vcounts, log_fcounts;
        printShape("log metab", log_metabolites);
        
        VectorXd metabolites = log_metabolites.array().exp();
        VectorXd metabolites_recip = metabolites.array().pow(-1.0);

        //nrxn x metabolite_count <= component product from:  (metabolite_count x 1) * (nrxn x metabolite_count)
        // printShape("metab recip", matabolites_recip);
        printShape("S", S);
        printShape("metab recip", metabolites_recip);
        // MatrixXd S_recip_metab = metabolites_recip.transpose().array() * (-1.0 * S).array().rowwise();
        // MatrixXd S_recip_metab = metabolites_recip.colwise().replicate(S.rows()).array() * (-1.0 * S).array();
        auto MR = metabolites_recip.rowwise().replicate(S.rows()).transpose();
        auto _S = (-1. * S);
        printShape("MR", MR);
        printShape("_S", _S);

        MatrixXd S_recip_metab = MR.array() * _S.array();

        std::cout << "S_recip_metab shape: (" << S_recip_metab.rows()
            << ", " << S_recip_metab.cols() << ")\n";
        
        VectorXd log_Q_inv = -1.0 * ( (R * log_metabolites) + (P * log_metabolites));
        VectorXd log_Q = 1.0 * ( (P * log_metabolites) + (R * log_metabolites));

        VectorXd x(nrxns);

        printShape("x", x);

        for (int rxn=0; rxn < nrxns; rxn++){
            double Q_inv_rxn = exp(log_Q_inv(rxn));
            double ekq_f = E_Regulation(rxn) * Keq_constant(rxn) * Q_inv_rxn;
            
            double Q_rxn = exp(log_Q(rxn));
            double ekq_r = E_Regulation(rxn) * pow(Keq_constant(rxn),-1.0) * Q_rxn; 

            x(rxn) = ekq_f + ekq_r;
        }
   
        //nrxn x metabolite_count <= component (nrxn x 1 ) * (nrxn x metabolite_count), but not sure how cwiseProduce is working. 
        //maybe need to do (x.transpose()).cwiseProduce(S_recip_metab.transpose()).transpose()
        // MatrixXd y = x.cwiseProduct(S_recip_metab).transpose();
        // MatrixXd y = x.array() * S_recip_metab.transpose().array().cols();
        MatrixXd y = x.rowwise().replicate(S.cols()).array() * S_recip_metab.array();
        printShape("y", y);

        //metabolite_count x metabolite_count <= (metabolite_count x nrxn) * (nrxn x metabolite_count)
        fjac = (S.transpose()) * y;

        std::cout << fjac << "\n";
        std::cout << __func__ << " end\n";
        return 0;
    }

    // int inputs() const { return m_inputs; }
    // int values() const { return m_values; }
};


/*
 * Driver function for calculating the least squares with the
 * Levenberg-Marquardt method
 */
[[nodiscard]]
Eigen::VectorXd least_squares(
            Eigen::MatrixXd& S_mat,
            Eigen::MatrixXd& R_back_mat,
            Eigen::MatrixXd& P_mat, 
            Eigen::VectorXd& Keq_constant,
            Eigen::VectorXd& E_Regulation,
            Eigen::VectorXd& log_fcounts,
            Eigen::VectorXd& log_vcounts)
{
    lmder_functor functor(
            S_mat,
            R_back_mat,
            P_mat,
            Keq_constant,
            E_Regulation,
            log_fcounts);

    std::cout << "Passed functor initialization\n";

    Eigen::LevenbergMarquardt<lmder_functor> lm(functor);

    std::cout << "Passed eigen thing initialization\n";

    lm.minimize(log_vcounts);

    std::cout << "Passed minimization\n";

    return log_vcounts;
}

#endif
