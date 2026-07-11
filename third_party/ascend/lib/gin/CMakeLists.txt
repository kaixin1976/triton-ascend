option(TRITON_ASCEND_BUILD_GIN_RUNTIME
  "Build the experimental Triton-Ascend GIN host runtime bridge" ON)

if(TRITON_ASCEND_BUILD_GIN_RUNTIME)
  set(ASCEND_CANN_ROOT "" CACHE PATH "CANN toolkit root for Triton-Ascend GIN runtime")

  find_path(TRITON_ASCEND_ACL_INCLUDE_DIR
    NAMES acl/acl.h
    HINTS
      ${ASCEND_CANN_ROOT}/include
      ${ASCEND_CANN_ROOT}/aarch64-linux/include
      $ENV{ASCEND_HOME_PATH}/include
      $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
      $ENV{ASCEND_TOOLKIT_HOME}/include
      $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
  )

  find_library(TRITON_ASCEND_ASCENDCL_LIBRARY
    NAMES ascendcl
    HINTS
      ${ASCEND_CANN_ROOT}/lib64
      ${ASCEND_CANN_ROOT}/aarch64-linux/lib64
      ${ASCEND_CANN_ROOT}/aarch64-linux/devlib/linux/aarch64
      $ENV{ASCEND_HOME_PATH}/lib64
      $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
      $ENV{ASCEND_HOME_PATH}/aarch64-linux/devlib/linux/aarch64
      $ENV{ASCEND_TOOLKIT_HOME}/lib64
      $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64
      $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/devlib/linux/aarch64
  )

  if(NOT TRITON_ASCEND_ACL_INCLUDE_DIR OR NOT TRITON_ASCEND_ASCENDCL_LIBRARY)
    message(FATAL_ERROR
      "TRITON_ASCEND_BUILD_GIN_RUNTIME=ON requires acl/acl.h and libascendcl. "
      "Set ASCEND_CANN_ROOT or source the CANN set_env.sh first.")
  endif()

  add_library(triton_ascend_gin_runtime SHARED
    triton_ascend_gin_runtime.cpp
  )
  target_include_directories(triton_ascend_gin_runtime
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${TRITON_ASCEND_ACL_INCLUDE_DIR}
  )
  target_link_libraries(triton_ascend_gin_runtime
    PRIVATE
      ${TRITON_ASCEND_ASCENDCL_LIBRARY}
      ${CMAKE_DL_LIBS}
  )
  set_target_properties(triton_ascend_gin_runtime PROPERTIES
    OUTPUT_NAME triton_ascend_gin_runtime
  )
  if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(_triton_ascend_gin_runtime_output_dir
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../backends/ascend/lib/gin")
    file(MAKE_DIRECTORY "${_triton_ascend_gin_runtime_output_dir}")
    set_target_properties(triton_ascend_gin_runtime PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${_triton_ascend_gin_runtime_output_dir}"
    )
  endif()
endif()
