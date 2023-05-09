#=============================================================================
# CMake configuration file for tracked vehicle on SPH terrain.
#=============================================================================

if(NOT ENABLE_MODULE_IRRLICHT)
  return()
endif()

include_directories(${CH_IRRLICHT_INCLUDES})
set(COMPILER_FLAGS "${CH_CXX_FLAGS}  ${CH_IRRLICHT_CXX_FLAGS}")
set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")
set(LIBRARIES ChronoEngine ChronoEngine_irrlicht ChronoEngine_vehicle ChronoEngine_vehicle_irrlicht ChronoModels_vehicle)
list(APPEND LIBS "")

if(ENABLE_MODULE_OPENGL)
  include_directories(${CH_OPENGL_INCLUDES})
  list(APPEND LIBRARIES "ChronoEngine_opengl")
endif()

if(ENABLE_MODULE_PARDISO_MKL)
    include_directories(${CH_MKL_INCLUDES})
    set(COMPILER_FLAGS "${COMPILER_FLAGS} ${CH_MKL_CXX_FLAGS}")
    set(LINKER_FLAGS "${LINKER_FLAGS} ${CH_MKL_LINK_FLAGS}")
    list(APPEND LIBRARIES "ChronoEngine_pardisomkl")
endif()

#--------------------------------------------------------------

find_package(Torch QUIET)
if(NOT Torch_FOUND)
  message(STATUS "Torch package not found - disabling Polaris NN test")
  return()
endif()

find_package(TorchCluster QUIET)
if(NOT TorchCluster_FOUND)
  message(STATUS "TorchCluster package not found - disabling Polaris NN test")
  return()
endif()

find_package(TorchScatter QUIET)
if(NOT TorchScatter_FOUND)
  message(STATUS "TorchScatter package not found - disabling Polaris NN test")
  return()
endif()

message(STATUS "TorchScatter_INCLUDE_DIR ${TorchScatter_INCLUDE_DIR}")
message(STATUS "TorchCluster_INCLUDE_DIR ${TorchCluster_INCLUDE_DIR}")


#--------------------------------------------------------------

set(PROGRAMS 
    test_Polaris_SCM
    test_Polaris_SCM_CustomTerrain
    )
    
set(UTILITIES
    DataWriter.h
    DataWriter.cpp
    CreateObjects.h
    CreateObjects.cpp
    SCMTerrain_Custom.h
    SCMTerrain_Custom.cpp
)

foreach(PROGRAM ${PROGRAMS})
  message(STATUS "...add ${PROGRAM}")

  add_executable(${PROGRAM}  ${PROGRAM}.cpp ${UTILITIES})
  source_group(""  FILES ${PROGRAM}.cpp ${UTILITIES})

  set_target_properties(${PROGRAM} PROPERTIES
        FOLDER demos
        COMPILER_FLAGS "${COMPILER_FLAGS}"
        LINK_FLAGS "${LINKER_FLAGS}"
  )

  set_property(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
  target_include_directories(${PROGRAM} PRIVATE ${TorchScatter_INCLUDE_DIR})
  target_include_directories(${PROGRAM} PRIVATE ${TorchCluster_INCLUDE_DIR})
  target_link_libraries(${PROGRAM} ${LIBRARIES})
  target_link_libraries(${PROGRAM} ${TORCH_LIBRARIES})
  target_link_libraries(${PROGRAM} TorchScatter::TorchScatter)
  target_link_libraries(${PROGRAM} TorchCluster::TorchCluster)
endforeach()
