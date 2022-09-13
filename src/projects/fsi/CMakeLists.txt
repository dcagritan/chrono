message(STATUS "Projects: FSI...")

if(NOT ENABLE_MODULE_FSI)
  return()
endif()  


# List FSI demos that do not have other dependencies
SET(FSI_DEMOS 
    #test_SPH_wheel_drop
    test_SPH_cohesion
)

#--------------------------------------------------------------

include_directories(${CH_FSI_INCLUDES})

set(COMPILE_FLAGS "${CH_CXX_FLAGS}")
set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")
set(LIBRARIES ChronoEngine ChronoEngine_fsi ChronoEngine_vehicle ChronoModels_vehicle)

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

MESSAGE(STATUS "Demo programs for FSI module...")

FOREACH(PROGRAM ${FSI_DEMOS})
    MESSAGE(STATUS "...add ${PROGRAM}")

  add_executable(${PROGRAM}  ${PROGRAM}.cpp)
  source_group(""  FILES ${PROGRAM}.cpp)

  set_target_properties(${PROGRAM} PROPERTIES
        FOLDER demos
        COMPILE_FLAGS "${COMPILE_FLAGS}"
        LINK_FLAGS "${LINKER_FLAGS}"
  )
  set_property(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
  target_link_libraries(${PROGRAM} ${LIBRARIES})

ENDFOREACH()
