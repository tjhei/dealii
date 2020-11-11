## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2020 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

#
# Configuration for thread support in deal.II with the help of the Taskflow
# library:
#


MACRO(FEATURE_TASKFLOW_FIND_EXTERNAL var)
  FIND_PACKAGE(TASKFLOW)

  IF(TASKFLOW_FOUND)
    SET(${var} TRUE)
  ENDIF()

  IF(TASKFLOW_VERSION VERSION_LESS "2.7")
    # Clear the previously determined version numbers to avoid confusion
    SET(TASKFLOW_VERSION "bundled")
    SET(TASKFLOW_VERSION_MAJOR "")
    SET(TASKFLOW_VERSION_MINOR "")

    MESSAGE(STATUS
      "The externally provided Taskflow library is older than version 2.7, "
      "which cannot be used with deal.II."
      )
    SET(TASKFLOW_ADDITIONAL_ERROR_STRING
      "The externally provided Taskflow library is older than version\n"
      "2.7, which is the oldest version compatible with deal.II."
      )
    SET(${var} FALSE)
  ENDIF()
ENDMACRO()


MACRO(FEATURE_TASKFLOW_CONFIGURE_BUNDLED)
  LIST(APPEND TASKFLOW_BUNDLED_INCLUDE_DIRS ${TASKFLOW_FOLDER}/include)
ENDMACRO()


CONFIGURE_FEATURE(TASKFLOW)

