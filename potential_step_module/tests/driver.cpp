#include <string>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra>
#include <filesystem>
using fs = std::filesystem;
using real = double;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

#include "helper_functions.hpp"

[[nodiscard]]
static inline std::vector<real>
read_vector(Vec& v, const std::string& path)
{
    std::vector v;
    std::ifstream f(path);
    while (!f.eof())
    {
        real d;
        f >> d;
        v.push_back(d);
    }
    return v;
}

int main(int argc, char** argv)
{
    fs::directory_entry datadir("data");
    if (!datadir.is_directory())
    {
        std::cout << "--- Expecting matrix and vector files to be in `data` directory.\n";
        return 1;
    }

    for (const auto& fn : { "Keq_constant.txt"
                            "P_mat.mtx"
                            "R_mat.mtx"
                            "S_mat.mtx"
                            "f_log_counts.txt"
                            "results.mtx"
                            "f_log_counts.txt"
                            "v_log_counts.txt" })
    {
        std::cout << "---- Searching for " << fn << " in " << datadir << "\n";
        std::ifstream f((datadir + fn).c_str());
        if (!f.good())
        {
            std::cout << "Didn't find " << fn << "\n";
            return 1;
        }
    }
    
    Mat S_mat;
    Mat S_mat;
    Mat S_mat;
    Mat results;
    Vec Keq_constant
    Vec f_log_counts;
    Vec v_log_counts;

    Eigen::loadMarket(P_mat, "P_mat.mtx");
    Eigen::loadMarket(R_matA, "R_mat.mtx");
    Eigen::loadMarket(S_matA, "S_mat.mtx");
    Eigen::loadMarket(results, "results.mtx");

    Vec Keq_constant( read_vector(datadir + "Keq_constant.txt").data() );
    Vec f_log_counts( read_vector(datadir + "f_log_counts.txt").data() );
    Vec v_log_counts( read_vector(datadir + "v_log_counts.txt").data() );

    std::cout << "--- Calling least squares\n"

    Vec result = least_squares(
            S_mat,
            R_back_mat,
            P_mat, 
            Keq_constant,
            E_Regulation,
            log_fcounts,
            log_vcounts);

    std::cout << "--- Successful least squares call.\n"
        << "--- Result: "
        << result;
}
