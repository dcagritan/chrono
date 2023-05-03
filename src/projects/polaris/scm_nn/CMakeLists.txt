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

set(PROGRAMS 
    test_Polaris_SCM_CustomTerrain
    )
    
set(UTILITIES
    DataWriter.h
    DataWriter.cpp
    CreateObjects.h
    CreateObjects.cpp
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
  target_link_libraries(${PROGRAM} ${LIBRARIES})
endforeach()
