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

set_target_properties(
    MonteLight PROPERTIES
    currentDir "${CMAKE_SOURCE_DIR}/build")

# TODO: Add tests and install targets if needed.
include(ExternalProject)

ExternalProject_Add(
    rapidjson
    PREFIX "${CMAKE_SOURCE_DIR}/external_includes/rapidjson"
    GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
    GIT_TAG f54b0e47a08782a6131cc3d60f94d038fa6e0a51
    TIMEOUT 10
    CMAKE_ARGS
        -DRAPIDJSON_BUILD_TESTS=OFF
        -DRAPIDJSON_BUILD_DOC=OFF
        -DRAPIDJSON_BUILD_EXAMPLES=OFF
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

# Prepare RapidJSON (RapidJSON is a header-only library)
ExternalProject_Get_Property(rapidjson source_dir)
set(RAPIDJSON_INCLUDE_DIR ${source_dir}/include)


include_directories(${RAPIDJSON_INCLUDE_DIR})