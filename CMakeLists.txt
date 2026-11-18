cmake_minimum_required(VERSION 3.20)
project(fishhash_poc C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Native code for Ada and Blackwell consumer cards.
set(CMAKE_CUDA_ARCHITECTURES 89 120)

if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /W3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /EHsc /W3")
  # Keep nvcc output quiet; errors still surface.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
endif()

# Vendored C libraries from iron-fish/fish-hash (BLAKE3 + Keccak-512).
# blake3.c is self-contained (includes portable.c and dispatch.c itself);
# its blake3_neon.c include is guarded by BLAKE3_USE_NEON (autodetected to
# 0 on x86, matching upstream BLAKE3).
add_library(fishhash_c STATIC
  third_party/blake3.c
  third_party/keccak.c)
target_include_directories(fishhash_c PUBLIC third_party)

# U1: test-vector generator (oracle harness).
add_executable(vecgen tools/vecgen.cpp)
target_link_libraries(vecgen PRIVATE fishhash_c)

# U2: CPU port of FishHash.
add_library(fishhash STATIC src/fishhash.cpp)
target_link_libraries(fishhash PUBLIC fishhash_c)
target_include_directories(fishhash PUBLIC src)

# U2: oracle gate against tests/vectors/vectors.json.
add_executable(test_fishhash tests/test_fishhash.cpp)
target_link_libraries(test_fishhash PRIVATE fishhash)

# U3: CUDA kernels (compiled by nvcc) + host-side GPU context.
find_package(CUDAToolkit REQUIRED)
add_library(fishhash_cuda STATIC src/kernels/fishhash.cu)
target_include_directories(fishhash_cuda PUBLIC third_party)
target_link_libraries(fishhash_cuda PUBLIC CUDA::cudart)

add_library(fishhash_gpu STATIC src/gpu_context.cpp)
target_link_libraries(fishhash_gpu PUBLIC fishhash_cuda fishhash)
target_include_directories(fishhash_gpu PUBLIC src)

# U3: GPU oracle gate (dataset + hash match vs CPU/JSON, planted solution).
add_executable(test_gpu_match tests/test_gpu_match.cpp)
target_link_libraries(test_gpu_match PRIVATE fishhash_gpu)

add_executable(fishbench tools/bench.cpp)
target_link_libraries(fishbench PRIVATE fishhash_gpu)

# U4: fishminer — node RPC client + batch-kernel GPU miner.
add_executable(fishminer src/main.cpp src/rpc_client.cpp)
target_link_libraries(fishminer PRIVATE fishhash_gpu)
if(WIN32)
  target_link_libraries(fishminer PRIVATE ws2_32)
endif()

enable_testing()
add_test(NAME fishhash COMMAND test_fishhash ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/vectors.json)
