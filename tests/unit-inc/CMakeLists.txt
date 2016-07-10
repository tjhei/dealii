CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)
INCLUDE(../setup_testsubproject.cmake)
PROJECT(testsuite CXX)

FILE(GLOB _ccs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
  )


ADD_EXECUTABLE(catcher
  ${_ccs}
  catch.hpp
  testing.h
  )
DEAL_II_SETUP_TARGET(catcher)

ADD_CUSTOM_TARGET(setup_tests
        COMMAND catcher --list-test-names-only >testlist || true
        COMMAND ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}
        DEPENDS catcher
        )

ENABLE_TESTING()

IF(EXISTS testlist)
  FILE(STRINGS testlist tests)
  FOREACH(test ${tests})
    STRING(REPLACE " " "_" testname ${test})
    MESSAGE("adding test ${testname}")
    ADD_TEST( ${testname} catcher ${test})
  ENDFOREACH()
ELSE()
  MESSAGE(" please run 'make setup_tests' to discover unit tests")
ENDIF()

