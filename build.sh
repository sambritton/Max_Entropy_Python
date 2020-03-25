#!/usr/bin/env bash

BUILDDIR=build
PYTHON=python3

proj_root=$(pwd)

echo Updating submodules...
git submodule init
git submodule update
module load python/3.7.0
module load cmake

echo Installing Eigen headers...
if [ -d ./potential_step_module/include/Eigen ]; then
    echo Headers already installed.
else
    cp -r ./eigen3/Eigen ./potential_step_module/include
fi

echo Checking pybind11 installation...
if [ ! -d pybind11/build ]
then
    mkdir ./pybind11/build
    pushd ./pybind11/build
    cmake ..
    if [ ! $? -eq 0 ]
    then
        echo Error building pybind11...
        exit 1
    fi
    popd
fi

if [ -d $BUILDDIR ]
then
    echo Removing old builds...
    rm -rf $BUILDDIR
fi

echo Building...
mkdir $BUILDDIR
pushd $BUILDDIR
cmake ..

if [ ! $? -eq 0 ]
then
    echo CMake error... Please check CMake output logs.
    exit 1
fi

echo Compiling...
make

if [ ! $? -eq 0 ]
then
    echo Make error... Please check configuration.
    exit 1
fi

echo Running tests...
cp $proj_root/potential_step_module/tests/*.py ./potential_step_module

echo
echo -----------TESTING-----------
echo

for _test in $(ls potential_step_module/*.py)
do
    $PYTHON ./$_test 2>&1 > $_test".log"
    if [ ! $? -eq 0 ]; then
        echo -- Test $(basename $_test) failed! See "$_test".log for output.
    else
        echo -- Test $_test passed
    fi
done

echo
echo -----------DONE TESTING-----------
echo

popd
