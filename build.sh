#!/usr/bin/env bash

BUILDDIR=build
BUILD=1
TEST=1
PROJ_ROOT=$(pwd)
MODDIR=$PROJ_ROOT/potential_step_module
OUTPUTDIR="$PROJ_ROOT/$BUILDDIR/potential_step_module"
n_jobs=1
PYTHON_EXE=""

while [[ $# -gt 0 ]]
do
    case $1 in
        -c|--compile-only)
            TEST=0
            shift
            ;;
        -t|--test-only)
            BUILD=0
            shift
            ;;
        -n|--num-jobs)
            n_jobs=$2
            shift; shift
            ;;
        -p|--python)
            PYTHON_EXE="$2"
            shift; shift
            ;;
        '-h'|'--help')
            ;&
        *)
            echo
            echo Usage:
            echo To only compile:
            printf '\t\t-c|--compile-only'
            echo
            echo To only run tests:
            printf '\t\t-t|--test-only'
            echo
            echo Number of make jobs:
            printf '\t\t-n|--num-jobs <int>'
            echo
            echo Python executable:
            printf '\t\t--python <path>'
            echo
            echo Help:
            printf '\t\t-h|--help'
            echo
            echo
            exit 1
            ;;
    esac
done

function error() {
    echo
    echo "Got error< $1 >"
    echo
    exit 1
}

[ -z "$PYTHON_EXE" ] && \
{
    printenv PYTHON_EXE 2>&1 >/dev/null || \
    {

        which python3 2>&1 >/dev/null ||
            error 'No python installation found on path...'
        # User has not defined a python executable...
        # just use the first python on the path
        PYTHON_EXE=$(which python3)
    }
}

PYTHON_VERSION=$($PYTHON_EXE --version 2>&1)
$PYTHON_EXE $MODDIR/tests/is_py3.py ||
	    error "Python installation found by script ($PYTHON_VERSION) is less than 3.5"

echo
echo Updating submodules...
echo

git submodule init
git submodule update

echo
echo Removing old headers
echo
rm -rf "$MODDIR/include/*"

if [ "$BUILD" -eq "1" ]; then
    echo
    echo Checking for needed programs...
    echo

    type module 2>&1 >/dev/null && {
        echo
        echo Loading modules...
        echo
        module load python/3.7.0
        module load cmake
        module load gcc/9.1.0
    }

    echo
    echo Installing Eigen headers...
    echo

    {
        pushd eigen3
        git remote remove origin
        git remote add origin https://gitlab.com/libeigen/eigen.git
        git pull
        git checkout 3.3
        git pull

        [ -d "$MODDIR/include" ] || mkdir "$MODDIR/include"
        cp -r ./Eigen/ "$MODDIR/include/Eigen"
        cp -r ./unsupported/ "$MODDIR/include/unsupported" || error 'installing eigen3'
        popd
    } &

    echo
    echo Checking pybind11 installation...
    echo

    {
        [ ! -d "$MODDIR/include/pybind11" ] && {
            cp -r pybind11/include/pybind11 $MODDIR/include
        }
    } &

    {
        if [ -d $BUILDDIR ]
        then
            echo
            echo Removing old builds...
            echo
            rm -rf $BUILDDIR
        fi
    } &

    wait

    echo
    echo Building...
    echo
    mkdir $BUILDDIR
    pushd $BUILDDIR
    cmake .. || error 'Could not configure bindings with cmake'

    echo
    echo Compiling...
    echo
    make -j $n_jobs || error 'Could not compile bindings'
    popd
fi

if [ "$TEST" -eq "1" ]; then
    echo
    echo Preparing tests...
    echo
    
    echo
    echo Cleaning up from previous builds if stale logs exist...
    echo
    rm $OUTPUTDIR/*.py 2>&1 > /dev/null
    rm $OUTPUTDIR/*.log 2>&1 > /dev/null

    echo
    echo Preparing newest version of tests...
    echo
    [ -d $OUTPUTDIR/test_cases ] || mkdir $OUTPUTDIR/test_cases
    cp $MODDIR/tests/test_*.py $OUTPUTDIR
    cp $MODDIR/tests/test_cases/*.p $OUTPUTDIR/test_cases/

    echo
    echo -----------TESTING-----------
    echo

    FAIL=0

    pushd $OUTPUTDIR 2>&1 >/dev/null
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
        echo -- Logs can be found in $OUTPUTDIR
        echo
        exit 1
    else
        echo -- SUMMARY
        echo -- Done! All tests passed.
        echo 

        if [ $(ls $OUTPUTDIR/*.py | wc -l) -gt 0 ]; then
            rm $OUTPUTDIR/*.py
            rm $OUTPUTDIR/*.log
        fi

        exit 0
    fi

fi

exit 0
