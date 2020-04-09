

//https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
struct lmder_functor : DenseFunctor<double>
{
    MatrixXd S;
    MatrixXd R;
    MatrixXd P;
    VectorXd Keq_constant;
    VectorXd E_Regulation;
    VectorXd log_fcounts;

    lmder_functor(
        MatrixXd& _S,
        MatrixXd& _R, 
        MatrixXd& _P,
        VectorXd& _Keq_constant,
        VectorXd& _E_Regulation,
        VectorXd& _log_fcounts):
        S(_S),
        R(_R),
        P(_P),
        Keq_constant(_Keq_constant),
        E_Regulation(_E_Regulation),
        log_fcounts(_log_fcounts) {}
   
    
    int operator()(const VectorXd& log_vcounts, VectorXd& deriv) const
    {
        //this function should be derivs
    
        int nrxns = S.rows();
        int nvar = log_vcounts.rows();//make sure this is length and not 1
        int metabolite_count = S.cols();
        

        VectorXd log_metabolites = (log_vcounts,log_fcounts); //should be constructed from vectors concatonated together. Not sure how to do that. 

        VectorXd log_Q_inv = -1.0 * ( (R * log_metabolites) + (P * log_metabolites));
        VectorXd log_Q = 1.0 * ( (P * log_metabolites) + (R * log_metabolites));

        VectorXd EKQ_f(nrxns);  //allocate. can break down to one vector but leave as two for clarity right now. 
        VectorXd EKQ_r(nrxns);    

        for (int rxn=0; rnx < nrxns; rxn++){
            double Q_inv_rxn = exp(log_Q_inv(rxn));
            double ekq_f = E_Regulation(rxn) * Keq_constant(rxn) * Q_inv_rxn;
            
            EKQ_f(rxn) = ekq_f;

            double Q_rxn = exp(log_Q(rxn));
            double ekq_r = E_Regulation(rxn) * pow(Keq_constant(rxn),-1.0) * Q_rxn; 
            EKQ_r(rxn) = ekq_r;
        }

        S.resize(nrxns,nvar); //take all rows (reactions) and only variable columns.

        //(nvar x 1) <=(nvar x nrxns) * (nrxns x 1)
        deriv = (S.transpose()) * (EKQ_f - EKQ_r);
    return 0;
    }

    
    //WARNING  jacobian should be calculated wrt metabolite concentration, not log(concentration). 
    int df(const VectorXd &log_vcounts, MatrixXd &fjac) const {
        //this should be a numerical jacobian
        //Jac is the Jacobian matrix, 
        //an N metabolite time-differential equations by (rows) by 
        //N metabolites derivatives (columns)
        //J_ij = d/dx_i(df_j/dt)

        int nrxns = S.rows();
        int nvar = log_vcounts.rows();//make sure this is length and not 1
        int metabolite_count = S.cols();

        //WARNING Only use to calcualte KQ
        VectorXd log_metabolites = (log_vcounts,log_fcounts); //should be constructed from vectors concatonated together. Not sure how to do that. 
        
        VectorXd metabolites = exp(log_metabolites); //should be constructed from vectors concatonated together. Not sure how to do that. 
        VectorXd metabolites_recip = pow(metabolites, -1.0);

        //nrxn x metabolite_count <= component product from:  (metabolite_count x 1) * (nrxn x metabolite_count)
        VectorXd S_recip_metab = metabolite_count.cwiseProduct( (-1.0 * S) );
        
        VectorXd log_Q_inv = -1.0 * ( (R * log_metabolites) + (P * log_metabolites));
        VectorXd log_Q = 1.0 * ( (P * log_metabolites) + (R * log_metabolites));

        VectorXd x(nrxns);

        for (int rxn=0; rnx < nrxns; rxn++){
            double Q_inv_rxn = exp(log_Q_inv(rxn));
            double ekq_f = E_Regulation(rxn) * Keq_constant(rxn) * Q_inv_rxn;
            
            double Q_rxn = exp(log_Q(rxn));
            double ekq_r = E_Regulation(rxn) * pow(Keq_constant(rxn),-1.0) * Q_rxn; 

            x(rxn) = ekq_f + ekq_r;
        }
   
        //nrxn x metabolite_count <= component (nrxn x 1 ) * (nrxn x metabolite_count), but not sure how cwiseProduce is working. 
        //maybe need to do (x.transpose()).cwiseProduce(S_recip_metab.transpose()).transpose()
        MatrixXd y = x.cwiseProduct(S_recip_metab).transpose();

        //metabolite_count x metabolite_count <= (metabolite_count x nrxn) * (nrxn x metabolite_count)
        fjac = (S.transpose()) * y;

        return 0;
    }
};