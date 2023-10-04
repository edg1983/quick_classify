nimble build -d:release -d:openmp -d:blas=openblas -d:lapack=openblas
mv quick_classify bin/

#nimble build -d:release -d:openmp -d:blas=openblas -d:lapack=openblas -d:avx512
#mv quick_classify bin/quick_classify_avx512
