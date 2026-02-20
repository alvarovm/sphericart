export Kokkos_DIR=/home/vama/soft/chem2/mace/kokkos/install
cmake .. \
 -DCMAKE_CXX_COMPILER=icpx\
 -DKokkos_DEVICES=SYCL\
 -DCMAKE_C_COMPILER=icx\
 -DCMAKE_CXX_FLAGS=" -qopenmp --intel -fsycl  -fsycl-targets=spir64 -Wno-deprecated-declarations -Wno-macro-redefined -Wno-unused-parameter -w"\
 -DSPHERICART_ENABLE_CUDA=OFF\
 -DSPHERICART_BUILD_TESTS=ON\
 -DOpenMP_CXX_FLAGS="-qopenmp"\
 -DSPHERICART_ENABLE_CUDA=OFF \
 -DSPHERICART_ENABLE_SYCL=ON \
 -DSPHERICART_SYCL_DEVICE=cpu\
 -DSPHERICART_PRINT_DEBUG=OFF\
 -DSPHERICART_SYCL_DTYPE=float\
 -DSPHERICART_BUILD_EXAMPLES=ON\
 -DSPHERICART_OPENMP=ON
 #-DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/2025.0/lib/libiomp5.a\
# -DOpenMP_CXX_LIB_NAMES="libiomp5"\
#-DSPHERICART_ENABLE_CUDA=OFF\
 #-DCMAKE_CXX_FLAGS=" -qopenmp --intel -fsycl -fsycl-targets=spir64 -Wno-deprecated-declarations -Wno-macro-redefined -Wno-unused-parameter -w"\

