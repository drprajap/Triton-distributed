#!/bin/bash

set -e

if [ -z $1 ]
then
  install_path=~/rocshmem
else
  install_path=$1
fi

src_path=$(dirname "$(realpath $0)")/../../../3rdparty/rocshmem/

cmake \
    -DBUILD_CODE_COVERAGE=${CODE_COV:-OFF} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-Release} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-~/rocshmem} \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_FUNCTIONAL_TESTS=ON \
    -DBUILD_UNIT_TESTS=ON \
    -DDEBUG=OFF \
    -DPROFILE=OFF \
    -DUSE_GDA=ON \
    -DUSE_RO=OFF \
    -DUSE_IPC=OFF \
    -DUSE_THREADS=OFF \
    -DUSE_WF_COAL=OFF \
    -DUSE_HDP_FLUSH=OFF \
    -DUSE_HDP_FLUSH_HOST_SIDE=OFF \
    -DGDA_MLX5=ON \
    $src_path
cmake --build . --parallel
cmake --install .
