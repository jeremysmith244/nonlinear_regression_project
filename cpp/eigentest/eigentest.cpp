#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::all;
 
int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  MatrixXd j(2,2);
  j = m;
  j(0,0) = 4;
  std::cout << m << std::endl;
  std::cout << j << std::endl;
  std::cout << m.cols() << std::endl;
  std::cout << "This is a slice" << std::endl;
  std::cout << m(all,1) << std::endl;
  std::cout << "This is mean" << std::endl;
  std::cout << m.mean() << std::endl;
  std::cout << "Power baby" <<std::endl;
  std::cout << m.array().pow(2) << std::endl;
}