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
        --test-only)
            BUILD=0
            ;;
        *)
            echo
            echo Usage:
            echo To only compile:
            printf '\t\t--compile-only'
            echo
            echo To only run tests:
            printf '\t\t--test-only'
            echo
            echo
            exit 1
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

if [ "$BUILD" -eq "1" ]; then
    echo
    echo Checking for needed programs...
    echo

    type module 2>&1 >/dev/null
    if [ $? -eq 0 ]; then
        echo
        echo Loading modules...
        echo
        module load python/3.7.0
        module load cmake
        module load gcc/9.1.0
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
    popd
fi

if [ "$TEST" -eq "1" ]; then
    echo
    echo Preparing tests...
    echo
    
    echo
    echo Cleaning up from previous builds if stale logs exist...
    echo
    rm $PROJ_ROOT/$BUILDDIR/potential_step_module/*.py 2>&1 > /dev/null
    rm $PROJ_ROOT/$BUILDDIR/potential_step_module/*.log 2>&1 > /dev/null

    echo
    echo Preparing newest version of tests...
    echo
    cp $PROJ_ROOT/potential_step_module/tests/test_*.py $PROJ_ROOT/$BUILDDIR/potential_step_module

    echo
    echo -----------TESTING-----------
    echo

    FAIL=0

    pushd $PROJ_ROOT/$BUILDDIR/potential_step_module 2>&1 >/dev/null
    for _test in $(ls ./test_*.py)
    do
        $PYTHON_EXE ./$_test 2>&1 > $_test".log"
        if [ ! $? -eq 0 ]; then
            echo -- Test $(basename $_test) failed!
            echo
            echo Output:
            cat ./$_test".log"
            echo
            FAIL=1
        else
            echo -- Test $_test passed
        fi
    done
    popd 2>&1 >/dev/null

    echo
    echo -----------DONE TESTING-----------
    echo

    if [ "$FAIL" -eq "1" ]; then
        echo -- SUMMARY
        echo -- Tests failed...
        echo -- Logs can be found in $PROJ_ROOT/$BUILDDIR/potential_step_module
        echo
        exit 1
    else
        echo -- SUMMARY
        echo -- Done! All tests passed.
        echo 

        if [ $(ls $PROJ_ROOT/$BUILDDIR/potential_step_module/*.py | wc -l) -gt 0 ]; then
            rm $PROJ_ROOT/$BUILDDIR/potential_step_module/*.py
            rm $PROJ_ROOT/$BUILDDIR/potential_step_module/*.log
        fi

        exit 0
    fi

fi

exit 0
