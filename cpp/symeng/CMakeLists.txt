cmake_minimum_required(VERSION 3.16)

project (symengtest)
 
find_package(SymEngine CONFIG)

set(CMAKE_CXX_FLAGS_RELEASE ${SYMENGINE_CXX_FLAGS_RELEASE})

include_directories(${SYMENGINE_INCLUDE_DIRS})
 
add_executable(symengtest symeng.cpp)

target_link_libraries(symengtest ${SYMENGINE_LIBRARIES})