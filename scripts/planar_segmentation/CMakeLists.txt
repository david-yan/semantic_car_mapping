cmake_minimum_required(VERSION 3.0.2)
project(planar_segmentation)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# set(Boost_PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libboost_python38.so)
# set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)
# set(Boost_INCLUDE_DIR /usr/include/)
# set(Boost_USE_STATIC_LIBS OFF)
# set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/share/cmake-3.16/Modules/)


# find_package(Boost REQUIRED COMPONENTS python3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)
find_package(PCL 1.2 REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(pickle REQUIRED)
# find_package(Boost REQUIRED COMPONENTS python)
find_package(jsoncpp REQUIRED)

## Find the directory containing pyconfig.h
execute_process(COMMAND python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES planar_segmentation
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
# ${Boost_INCLUDE_DIRS}
${PYTHON_INCLUDE_DIR}
)

add_executable (planar_segmentation planar_segmentation.cpp)

target_link_libraries (planar_segmentation 
${PCL_LIBRARIES} 
${EIGEN3_LIBS} 
# ${Boost_LIBRARIES}
${Jsoncpp_LIBRARIES}
jsoncpp_lib
# ${pickle_LIBRARIES}
)
