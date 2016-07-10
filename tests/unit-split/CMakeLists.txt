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

