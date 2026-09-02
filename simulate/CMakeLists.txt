cmake_minimum_required(VERSION 3.16)
project(unitree_mujoco)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

option(BUILD_P1_MUJOCO_DEPLOY_SIM "Build the P1 SDK-free single-process MuJoCo policy runner" ON)
option(P1_ENABLE_BUILTIN_POLICY_DEFAULTS
  "Allow hard-coded P1 policy parameters when deploy.yaml is not loaded" OFF)
get_filename_component(P1_DEFAULT_ROBOT_DEPLOY_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../robot_deploy-main" ABSOLUTE)
set(P1_ROBOT_DEPLOY_ROOT "${P1_DEFAULT_ROBOT_DEPLOY_ROOT}" CACHE PATH
  "Path to robot_deploy-main for TorchPolicyRunner sources")


include_directories(
  mujoco/include
)
link_directories(mujoco/lib)

add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)
target_link_libraries(jstest PRIVATE pthread)

add_executable(p1_ankle_kinematics_check
  src/ankle_kinematics_check.cpp
)

if(BUILD_P1_MUJOCO_DEPLOY_SIM)
  set(P1_TORCH_PREFIX_HINTS)
  set(P1_SKIP_TORCH_FIND_PACKAGE OFF)
  set(P1_TORCH_READY OFF)
  set(P1_TORCH_CUDA_RESULT 1)
  set(P1_TORCH_CUDA_VERSION "")
  set(P1_PYTORCH_TORCH_DIR "")
  set(P1_TORCH_INCLUDE_DIRS)
  set(P1_TORCH_LINK_OPTIONS)
  set(P1_TORCH_LIBS)
  set(P1_TORCH_COMPILE_OPTIONS)
  set(P1_LOCAL_LIBTORCH_DIR "${P1_ROBOT_DEPLOY_ROOT}/src/inference/third_party/libtorch")
  set(P1_LOCAL_PYTORCH_DIR "${P1_ROBOT_DEPLOY_ROOT}/src/inference/third_party/python_torch/torch")

  if(EXISTS "${P1_LOCAL_LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND P1_TORCH_PREFIX_HINTS "${P1_LOCAL_LIBTORCH_DIR}")
  endif()

  if(EXISTS "${P1_LOCAL_PYTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND P1_TORCH_PREFIX_HINTS "${P1_LOCAL_PYTORCH_DIR}/share/cmake")
  endif()

  find_program(P1_PYTHON3_EXECUTABLE python3)
  if(P1_PYTHON3_EXECUTABLE)
    execute_process(
      COMMAND "${P1_PYTHON3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
      RESULT_VARIABLE P1_PYTORCH_PREFIX_RESULT
      OUTPUT_VARIABLE P1_PYTORCH_CMAKE_PREFIX
      ERROR_QUIET
    )
    if(P1_PYTORCH_PREFIX_RESULT EQUAL 0 AND EXISTS "${P1_PYTORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake")
      list(APPEND P1_TORCH_PREFIX_HINTS "${P1_PYTORCH_CMAKE_PREFIX}")
    endif()
    execute_process(
      COMMAND "${P1_PYTHON3_EXECUTABLE}" -c "import torch; print(torch.version.cuda or '', end='')"
      RESULT_VARIABLE P1_TORCH_CUDA_RESULT
      OUTPUT_VARIABLE P1_TORCH_CUDA_VERSION
      ERROR_QUIET
    )
    execute_process(
      COMMAND "${P1_PYTHON3_EXECUTABLE}" -c "import torch, os; print(os.path.dirname(torch.__file__), end='')"
      RESULT_VARIABLE P1_PYTORCH_TORCH_DIR_RESULT
      OUTPUT_VARIABLE P1_PYTORCH_TORCH_DIR
      ERROR_QUIET
    )
  endif()

  if(P1_TORCH_PREFIX_HINTS)
    list(APPEND CMAKE_PREFIX_PATH ${P1_TORCH_PREFIX_HINTS})
  endif()

  if(P1_TORCH_CUDA_RESULT EQUAL 0 AND P1_TORCH_CUDA_VERSION AND
     NOT DEFINED CUDA_TOOLKIT_ROOT_DIR AND
     NOT DEFINED CUDAToolkit_ROOT AND
     NOT EXISTS "/usr/local/cuda")
    message(STATUS
      "P1 Torch runners: Python torch is CUDA-enabled (${P1_TORCH_CUDA_VERSION}) "
      "but no CUDA toolkit root is visible to CMake; trying direct Python torch shared-library fallback.")
    set(P1_SKIP_TORCH_FIND_PACKAGE ON)
  endif()

  if(NOT P1_SKIP_TORCH_FIND_PACKAGE)
    find_package(Torch QUIET)
  endif()
  set(P1_HAS_LOCAL_TORCH_RUNNER OFF)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_policy_runner.hpp" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_policy_runner.cpp")
    set(P1_HAS_LOCAL_TORCH_RUNNER ON)
  endif()

  if(Torch_FOUND AND P1_HAS_LOCAL_TORCH_RUNNER)
    set(P1_TORCH_READY ON)
    set(P1_TORCH_LIBS ${TORCH_LIBRARIES})
    set(P1_TORCH_COMPILE_OPTIONS ${TORCH_CXX_FLAGS})
  elseif(P1_SKIP_TORCH_FIND_PACKAGE AND P1_HAS_LOCAL_TORCH_RUNNER AND
         EXISTS "${P1_PYTORCH_TORCH_DIR}/include/torch/script.h" AND
         EXISTS "${P1_PYTORCH_TORCH_DIR}/include/torch/csrc/api/include/torch/torch.h" AND
         EXISTS "${P1_PYTORCH_TORCH_DIR}/lib/libtorch_cpu.so" AND
         EXISTS "${P1_PYTORCH_TORCH_DIR}/lib/libtorch.so" AND
         EXISTS "${P1_PYTORCH_TORCH_DIR}/lib/libc10.so")
    set(P1_TORCH_READY ON)
    set(P1_TORCH_INCLUDE_DIRS
      ${P1_PYTORCH_TORCH_DIR}/include
      ${P1_PYTORCH_TORCH_DIR}/include/torch/csrc/api/include
    )
    set(P1_TORCH_LINK_OPTIONS
      -Wl,--no-as-needed
      -Wl,-rpath,${P1_PYTORCH_TORCH_DIR}/lib
    )
    set(P1_TORCH_LIBS
      ${P1_PYTORCH_TORCH_DIR}/lib/libtorch_cpu.so
      ${P1_PYTORCH_TORCH_DIR}/lib/libtorch.so
      ${P1_PYTORCH_TORCH_DIR}/lib/libc10.so
    )
  elseif(NOT P1_SKIP_TORCH_FIND_PACKAGE)
    message(WARNING
      "Skipping P1 Torch runners: Torch or local TorchPolicyRunner was not found.")
  else()
    message(WARNING
      "Skipping P1 Torch runners: direct Python torch fallback is not usable.")
  endif()

  if(P1_TORCH_READY)
    set(P1_POLICY_RUNTIME_SOURCES
      src/p1_config.cpp
      src/p1_policy_controller.cpp
      src/mujoco_p1_backend.cpp
      src/torch_policy_runner.cpp
    )

    add_executable(p1_mujoco_deploy_sim
      src/main.cpp
      ${P1_POLICY_RUNTIME_SOURCES}
      src/p1_sim_log.cpp
      src/mujoco_viewer.cpp
      src/elastic_rope.cpp
      src/joystick/p1_joystick_velocity.cc
      src/joystick/joystick.cc
    )
    target_compile_definitions(p1_mujoco_deploy_sim PRIVATE
      P1_ENABLE_BUILTIN_POLICY_DEFAULTS=$<BOOL:${P1_ENABLE_BUILTIN_POLICY_DEFAULTS}>
    )
    target_compile_options(p1_mujoco_deploy_sim PRIVATE ${P1_TORCH_COMPILE_OPTIONS})
    target_include_directories(p1_mujoco_deploy_sim PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${P1_ROBOT_DEPLOY_ROOT}/src/inference/include
      ${P1_TORCH_INCLUDE_DIRS}
    )
    if(P1_TORCH_LINK_OPTIONS)
      target_link_options(p1_mujoco_deploy_sim PRIVATE ${P1_TORCH_LINK_OPTIONS})
    endif()
    target_link_libraries(p1_mujoco_deploy_sim PRIVATE
      ${P1_TORCH_LIBS}
      mujoco
      glfw
      yaml-cpp
      fmt
      pthread
      dl
    )

    add_executable(real2sim_replay
      src/real2sim_replay.cpp
      ${P1_POLICY_RUNTIME_SOURCES}
    )
    target_compile_definitions(real2sim_replay PRIVATE
      P1_ENABLE_BUILTIN_POLICY_DEFAULTS=$<BOOL:${P1_ENABLE_BUILTIN_POLICY_DEFAULTS}>
    )
    target_compile_options(real2sim_replay PRIVATE ${P1_TORCH_COMPILE_OPTIONS})
    target_include_directories(real2sim_replay PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${P1_ROBOT_DEPLOY_ROOT}/src/inference/include
      ${P1_TORCH_INCLUDE_DIRS}
    )
    if(P1_TORCH_LINK_OPTIONS)
      target_link_options(real2sim_replay PRIVATE ${P1_TORCH_LINK_OPTIONS})
    endif()
    target_link_libraries(real2sim_replay PRIVATE
      ${P1_TORCH_LIBS}
      mujoco
      yaml-cpp
      fmt
      pthread
      dl
    )
  endif()
endif()
