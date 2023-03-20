#--------------------------------------------------------------
# Validation tests SMC method with Chrono::Multicore
#
# Requires the Multicore and Irrlicht modules 
#--------------------------------------------------------------

message(STATUS "Projects: SMC validation...")

if(NOT ENABLE_MODULE_VEHICLE)
  return()
endif()

if(NOT ENABLE_MODULE_IRRLICHT OR NOT ENABLE_MODULE_MULTICORE)
  return()
endif()  


set(TESTS
  test_SMC_cohesion  
  test_SMC_cor_normal
  test_SMC_cor_oblique
  test_SMC_rolling_cohesion
  test_SMC_rolling_gravity
  test_SMC_sliding_cohesion
  test_SMC_sliding_gravity
  test_SMC_spinning_cohesion
  test_SMC_spinning_gravity
  test_SMC_stacking
  )


include_directories(${CH_MULTICORE_INCLUDES})
include_directories(${CH_IRRLICHT_INCLUDES})

foreach(PROGRAM ${TESTS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp" test_SMC.h)
    source_group(""  FILES "${PROGRAM}.cpp" test_SMC.h)

    set_target_properties(${PROGRAM} PROPERTIES
        FOLDER tests
        COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_MULTICORE_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}"
    )

    target_link_libraries(${PROGRAM}
                          ChronoEngine
                          ChronoEngine_multicore
                          ChronoEngine_irrlicht)

endforeach(PROGRAM)

