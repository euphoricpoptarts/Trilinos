
SET(HEADERS "")
SET(SOURCES "")

SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.h)
APPEND_GLOB(SOURCES ${DIR}/*.C)

INCLUDE_DIRECTORIES(
  "${CMAKE_CURRENT_SOURCE_DIR}/../"
  "${CMAKE_CURRENT_BINARY_DIR}/../"
)

list(APPEND HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/iotm_export.h")

TRIBITS_ADD_LIBRARY(
	Iotm
	HEADERS	${HEADERS}
	SOURCES ${SOURCES}
	DEPLIBS Ioss
)
generate_export_header(Iotm)
target_include_directories(Iotm
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")


IF (BUILD_TESTING)
  ENABLE_TESTING()
ENDIF()
