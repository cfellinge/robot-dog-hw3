#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <random>

// Reference:
// https://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf

using Eigen::MatrixXf;

Eigen::MatrixXf reduceRank(Eigen::MatrixXf mat)
{
    // Perform SVD
    Eigen::BDCSVD<Eigen::MatrixXf> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Get the singular values
    Eigen::VectorXf singular_values = svd.singularValues();
    Eigen::MatrixXf u = svd.matrixU();
    Eigen::MatrixXf v = svd.matrixV();

    // set bottom right to 0
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SVDBase.html#a4e7bac123570c348f7ed6be909e1e474
    // Singular values are always in decreasing order
    v(v.rows()-1,v.cols() - 1) = 0;

    // Recompose the matrix
    MatrixXf f = u * singular_values.asDiagonal() * v.transpose();

    return f;
}

int main()
{
    srand(0);

    MatrixXf p(3, 8);
    MatrixXf p_prime(3, 8);

    MatrixXf transformation(3, 3);
    transformation.row(0) << 1, 4, 4;
    transformation.row(1) << 0, 4, 4;
    transformation.row(2) << 4, 0, 7;
    // transformation.row(0) << 1, 0, 0;
    // transformation.row(1) << 0, 1, 0;
    // transformation.row(2) << 0, 0, 1;

    p.row(0) << 1, -2, 2, 1, 3, 1, -1, 5;
    p.row(1) << 2, 3, -1, 0, -2, 1, 0, -3;
    p.row(2) = MatrixXf::Ones(1, 8);

    for (int i = 0; i < 8; i++)
    {
        p_prime.col(i) = p.col(i).transpose() * transformation;
    }

    p.conservativeResize(p.rows() - 1, p.cols());
    p_prime.conservativeResize(p_prime.rows() - 1, p_prime.cols());

    for (int i = 0; i < 8; i++)
    {
        p.col(i).normalize();
        p_prime.col(i).normalize();
    }

    p.conservativeResize(p.rows() + 1, p.cols());
    p_prime.conservativeResize(p_prime.rows() + 1, p_prime.cols());

    p.row(2) = MatrixXf::Ones(1, 8);
    p_prime.row(2) = MatrixXf::Ones(1, 8);

    std::cout << "p:\n"
              << p << std::endl << std::endl;
    std::cout << "p':\n"
              << p_prime << std::endl << std::endl;

    MatrixXf a(8, 9);

    for (int i = 0; i < 8; i++)
    {
        // x * x'
        a.row(i)[0] = p.col(i)[0] * p_prime.col(i)[0];

        // x * y'
        a.row(i)[1] = p.col(i)[0] * p_prime.col(i)[1];

        // x
        a.row(i)[2] = p.col(i)[0];

        // y * x'
        a.row(i)[3] = p.col(i)[1] * p_prime.col(i)[0];

        // y * y'
        a.row(i)[4] = p.col(i)[1] * p_prime.col(i)[1];

        // y
        a.row(i)[5] = p.col(i)[1];

        // x'
        a.row(i)[6] = p_prime.col(i)[0];

        // y'
        a.row(i)[7] = p_prime.col(i)[1];

        // 1
        a.row(i)[8] = 1;
    }
    std::cout << "\nA:" << std::endl;
    std::cout << a << std::endl;

    MatrixXf b(8, 1);
    b = MatrixXf::Zero(8, 1);

    std::cout << "\nb:" << std::endl;
    std::cout << b << std::endl << std::endl;

    // // Alternate step 2:

    MatrixXf aTa = a.transpose() * a;
    Eigen::BDCSVD<Eigen::MatrixXf> bdcsvd(aTa,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXf singular_values = bdcsvd.singularValues();

    // Find the smallest singular value
    int min_index;
    float min_value = singular_values.minCoeff(&min_index);

    std::cout << "\nsigma:" << std::endl;
    std::cout << singular_values << std::endl << std::endl;
    std::cout << "\nmin val/index:" << std::endl;
    std::cout << min_value << "/" << min_index << std::endl << std::endl;

    // F is v corresponding to least signular value
    MatrixXf f = bdcsvd.matrixV().col(min_index);

    std::cout << "\nf:" << std::endl;
    std::cout << f << std::endl << std::endl;

    // Put the 9x1 matrix into its 3x3 shape
    MatrixXf F(3, 3);
    for (int i = 0; i < 9; i++)
    {
        F.row(i / 3)[i % 3] = f.row(i)[0];
    }
    
    std::cout << "\nF:" << std::endl;
    std::cout << F << std::endl << std::endl;

    // Check that we have in fact minimized the thing (printed numbers should be really small)
    for (int i = 0; i< 8; i++)
    {
        std::cout << "\np" <<i << "T * F * p" <<i << "\':" << std::endl;
        std::cout << p.col(i).transpose() * F * p_prime.col(i) << std::endl << std::endl;
    }

    // Enforce rank 2
    MatrixXf Fp = reduceRank(F);

    std::cout << "\nFp:" << std::endl;
    std::cout << Fp << std::endl << std::endl;


    // Check that we have in fact minimized the thing (printed numbers should be really small)
    for (int i = 0; i< 8; i++)
    {
        std::cout << "\np" <<i << "T * Fp * p" <<i << "\':" << std::endl;
        std::cout << p.col(i).transpose() * Fp * p_prime.col(i) << std::endl << std::endl;
    }
}
