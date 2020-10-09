#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${QMCPACK_UNIT_TEST_DIR})

FOREACH(NAME math reduction target_task complex complex_reduction math_with_complex_header)
  SET(UTEST_EXE test_omptarget_${NAME})
  SET(UTEST_NAME unit_${UTEST_EXE})
  ADD_EXECUTABLE(${UTEST_EXE} test_omp_${NAME}.cpp)
  TARGET_LINK_LIBRARIES(${UTEST_EXE} platform_device catch_main ${MPI_LIBRARY})
  ADD_UNIT_TEST(${UTEST_NAME} "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
ENDFOREACH()