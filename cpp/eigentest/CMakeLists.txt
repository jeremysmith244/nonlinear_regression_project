cmake_minimum_required(VERSION 3.16)

project (eigentest)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
add_executable(eigentest eigentest.cpp)

target_link_libraries (eigentest Eigen3::Eigen)