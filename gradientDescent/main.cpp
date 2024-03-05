#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <random>
#include <iomanip>

using Eigen::MatrixXf;
using Eigen::VectorXf;

VectorXf findGradient(MatrixXf a, VectorXf x)
{
    // return a.transpose() * a * x - a.transpose() * b;
    return x.transpose() * (a + a.transpose());
}

float f(MatrixXf a, VectorXf x)
{
    return x.transpose() * a * x;
}

int main()
{
    MatrixXf a(2, 2);

    VectorXf x_0(2);
    VectorXf x(2);
    VectorXf x_prime(2);

    a.row(0) << 2, -1;
    a.row(1) << -1, 1;

    x_0 << 30, -30;

    x = x_0;

    int k = 0;
    float epsilon = 0.0001;
    float magnitude = 0;
    float alpha = 1;

    VectorXf gradient;

    const float C = 0.5;
    const float TAU = 0.5;

    do
    {
        // get gradient
        gradient = findGradient(a, x);
        // std::cout << "Grad: \n" << gradient << std::endl;
        // std::cout << "Norm: \n" << gradient.norm() << std::endl;

        // step 3
        x_prime = x - alpha * gradient;

        // step 4
        if (f(a, x_prime) > f(a, x))
        {
            // descent direction is p
            // descent direction is opposite of gradient
            VectorXf p(2);
            p = -1 * gradient;

            // m is the local slope
            float m = gradient.transpose() * p;

            // t is a magic number used in the for loop
            // combo of other stuff
            float t = -1 * C * m;

            // j is counter
            int j = 0;

            // Iterate until the jump is small enough to not
            // accidently jump over minimum
            while (f(a, x) - f(a, x + alpha * p) < alpha * t)
            {
                j++;
                // new smaller alpha
                alpha = TAU * alpha;
            }

            // new, better x_prime
            x_prime = x - alpha * gradient;
        }
        x = x_prime;
        k++;
    } while (gradient.norm() > epsilon);

    std::cout << "x (local min): \n" << std::fixed << std::setprecision(2) << x << std::endl;
}