#!/usr/bin/env bash

BUILDDIR=build
BUILD=1
TEST=1
PROJ_ROOT=$(pwd)
LOCAL_INSTALL=$PROJ_ROOT/potential_step_module

for i in "$@"
do
    case $i in
        --compile-only)
            TEST=0
            ;;
    esac
done

function error_check() {
    if [ ! $? -eq 0 ]; then
        echo
        echo "Got error< $1 >"
        echo
        exit 1
    fi
}

printenv PYTHON_EXE 2>&1 >/dev/null
if [ ! $? -eq 0 ]; then
    which python 2>&1 >/dev/null
    error_check 'No python installation found on path...'
    # User has not defined a python executable...
    # just use the first python on the path
    PYTHON_EXE=$(which python)
fi

PYTHON_VERSION=$($PYTHON_EXE --version 2>&1)
$PYTHON_EXE $PROJ_ROOT/potential_step_module/tests/is_py3.py
error_check "Python installation found by script ($PYTHON_VERSION) is less than 3.5"

echo
echo Updating submodules...
echo

git submodule init
git submodule update

echo
echo Checking for needed programs...
echo

type module 2>&1 >/dev/null
if [ $? -eq 0 ]; then
    module load python/3.7.0
    module load cmake
fi

echo
echo Installing Eigen headers...
echo

if [ -d "$LOCAL_INSTALL/include/Eigen" ]; then
    echo
    echo Headers already installed.
    echo
else

    cp -r eigen3/Eigen/ "$LOCAL_INSTALL/include/Eigen"
    error_check 'installing eigen3'

    popd
fi

echo
echo Checking pybind11 installation...
echo
if [ ! -d "$LOCAL_INSTALL/include/pybind11" ]
then
    cp -r pybind11/include/pybind11 $LOCAL_INSTALL/include/pybind11
fi

if [ -d $BUILDDIR ]
then
    echo
    echo Removing old builds...
    echo
    rm -rf $BUILDDIR
fi

echo
echo Building...
echo
mkdir $BUILDDIR
pushd $BUILDDIR
cmake ..
error_check 'Could not configure bindings with cmake'

echo
echo Compiling...
echo
make
error_check 'Could not compile bindings'

echo
echo Running tests...
echo
cp $PROJ_ROOT/potential_step_module/tests/*.py ./potential_step_module

if [ "$TEST" -eq "1" ]; then
    echo
    echo -----------TESTING-----------
    echo

    FAIL=0
    for _test in $(ls potential_step_module/*.py)
    do
        $PYTHON_EXE ./$_test 2>&1 > $_test".log"
        if [ ! $? -eq 0 ]; then
            echo -- Test $(basename $_test) failed!
            echo
            echo Output:
            cat ./$_test".log"
            FAIL=1
        else
            echo -- Test $_test passed
        fi
    done

    echo
    echo -----------DONE TESTING-----------
    echo
fi

popd

if [ "$FAIL" -eq "1" ]; then
    echo
    echo Tests failed...
    echo
    exit 1
else
    echo
    echo Done!
    echo 
    exit 0
fi
