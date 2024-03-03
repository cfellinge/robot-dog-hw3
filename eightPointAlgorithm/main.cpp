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
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get the singular values
    Eigen::VectorXf singular_values = svd.singularValues();

    // Find the smallest singular value and set it to zero
    int min_index;
    float min_value = singular_values.minCoeff(&min_index);
    singular_values(min_index) = 0.0;

    // Recompose the matrix
    Eigen::MatrixXf mat_rank2 = svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();

    // Print the original and rank-2 matrix
    std::cout << "\nOriginal matrix:\n"
              << mat << std::endl;
    std::cout << "\nRank-2 matrix:\n"
              << mat_rank2 << std::endl;

    return mat_rank2;
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
              << p << std::endl;
    std::cout << "p':\n"
              << p_prime << std::endl;

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

    // Alternate step 2:

    MatrixXf aTa = a.transpose() * a;

    Eigen::JacobiSVD<MatrixXf> svdOfaTa;
    svdOfaTa.compute(aTa, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Step 2:

    Eigen::JacobiSVD<MatrixXf> svdOfA;
    svdOfA.compute(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

    std::cout << "\nSingular values:" << std::endl;
    std::cout << svdOfaTa.singularValues() << std::endl;

    MatrixXf b(9, 1);
    b = MatrixXf::Ones(9, 1);

    // // solve Ax = b for x
    // MatrixXf x = svdOfA.solve(b);
    // // TODO: What is b

    // std::cout << "\nx*:" << std::endl;
    // std::cout << x << std::endl;

    // std::cout << "x* is a local optimum." << std::endl;

    // Perform SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get the singular values
    Eigen::VectorXf singular_values = svd.singularValues();

    // Find the smallest singular value and set it to zero
    int min_index;
    float min_value = singular_values.minCoeff(&min_index);
    MatrixXf e = svd.matrixV().col(min_index);

    std::cout << "y: " << e << std::endl;

    // Step 3:

    // MatrixXf e(3, 3);

    // for (int i = 0; i < 9; i++)
    // {
    //     e.row(i / 3)[i % 3] = x.row(i)[0];
    // }

    std::cout << "\nE*:" << std::endl;
    std::cout << e << std::endl;

    Eigen::FullPivLU<MatrixXf> lu_decomp(e);
    auto rank = lu_decomp.rank();

    std::cout << "Rank:" << rank << std::endl;

    std::cout << "pp" << std::endl;

    std::cout << e * (e.transpose() * e).inverse() * e.transpose() << std::endl;

    // Step 4:

    MatrixXf f = reduceRank(e);

    Eigen::FullPivLU<MatrixXf> flu_decomp(f);
    auto frank = flu_decomp.rank();

    std::cout << "Rank:" << frank << std::endl;

    std::cout << "\n{{";

    for (int i = 0; i < 9; i++)
    {
        std::cout << f.row(i / 3)[i % 3];
        if (i % 3 == 2 && i != 8)
        {
            std::cout << "}, {";
        }
        else if (i == 8)
        {
            break;
        }
        else
        {
            std::cout << ", ";
        }
    }

    std::cout << "}}\n"
              << std::endl;

    for (int i = 0; i < 8; i++)
    {
        float diff = p_prime.col(i).transpose() * f * p.col(i);
        std::cout << i + 1 << ": " << diff << std::endl;
    }

    std::cout << f * (f.transpose() * f).inverse() * f.transpose() << std::endl;
}
