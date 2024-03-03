#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>

using Eigen::MatrixXf;

Eigen::MatrixXf reduceRank(Eigen::MatrixXf mat) {
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
    std::cout << "\nOriginal matrix:\n" << mat << std::endl;
    std::cout << "\nRank-2 matrix:\n" << mat_rank2 << std::endl;

    return mat_rank2;
}

int main()
{
    MatrixXf m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);

    MatrixXf points(3, 10);
    points = MatrixXf::Random(3, 10);

    // Eight randomly generated point pairs
    MatrixXf p(2, 8);
    p = MatrixXf::Random(2, 8);
    p.conservativeResize(p.rows() + 1, p.cols());
    p.row(p.rows() - 1) = MatrixXf::Ones(1, 8);

    std::cout << "\np:" << std::endl;
    std::cout << p << std::endl;

    MatrixXf p_prime(2, 8);
    p_prime = MatrixXf::Random(2, 8);
    p_prime.conservativeResize(p_prime.rows() + 1, p_prime.cols());
    p_prime.row(p_prime.rows() - 1) = MatrixXf::Ones(1, 8);
    
    std::cout << "\np':" << std::endl;
    std::cout << p_prime << std::endl;

    MatrixXf a(8, 9);

    for (int i = 0; i < 8; i++) {
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

    // a = a.cast<float>();

    // MatrixXf x = Eigen::JacobiSVD(a);
    Eigen::JacobiSVD<MatrixXf> svd;
    svd.compute(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

    std::cout << "\nSingular values:" << std::endl;
    std::cout << svd.singularValues() << std::endl;

    MatrixXf b(8, 1);
    b = MatrixXf::Ones(8, 1);
    
    MatrixXf x = svd.solve(b);

    std::cout << "\nx*:" << std::endl;
    std::cout << x << std::endl;

    std::cout << "x* is a local optimum." << std::endl;

    MatrixXf e(3, 3);

    for (int i = 0; i < 9; i++) {
        e.row(i / 3)[i % 3] = x.row(i)[0];
    }

    std::cout << "\nE*:" << std::endl;
    std::cout << e << std::endl;

    reduceRank(e);
}
