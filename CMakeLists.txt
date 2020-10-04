# CMakeList.txt : CMake project for MonteLight, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("MonteLight")

# Add source to this project's executable.
add_executable (MonteLight "montelight.cc")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# TODO: Add tests and install targets if needed.
