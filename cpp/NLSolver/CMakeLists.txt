cmake_minimum_required(VERSION 3.16)

project (NLSolver)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
add_executable(NLSolver NLSolver.cpp)

target_link_libraries (NLSolver Eigen3::Eigen)