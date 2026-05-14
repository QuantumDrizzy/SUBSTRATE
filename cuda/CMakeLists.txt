cmake_minimum_required(VERSION 3.22)
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
# Target architectures: Volta(75) Ampere(86) Ada(89) Hopper(90) Blackwell(120)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90 120)

add_library(substrate_cuda STATIC
    radical_pair.cu
    tensor_net.cu
)

target_compile_options(substrate_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler=-fPIC
        -lineinfo
    >
)

set_target_properties(substrate_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
