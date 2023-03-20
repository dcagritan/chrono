if(NOT ENABLE_MODULE_IRRLICHT)
  return()
endif()  

# ------------------------------------------------------------------------------
# List of all executables

set(DEMOS
    test_VEH_bump
    test_VEH_Bezier
    test_VEH_BezierTracker
    test_VEH_steering_compliance
    test_VEH_steering_input
    test_VEH_ISO2631_shock_ramp
    test_VEH_DLC_hmmwv
    test_VEH_circle_hmmwv
    test_VEH_friction_hmmwv
    test_VEH_friction_variable_hmmwv
    test_VEH_friction_m113
    test_VEH_friction_variable_m113
    test_VEH_tire_csys
    test_VEH_SCM_hm    
    test_VEH_WheeledJSON_Accel
    test_VEH_HMMWV_acc_tst
)

if(THRUST_FOUND)
  set(DEMOS ${DEMOS}
      test_VEH_SIMD_test_m113
     )
endif()

#--------------------------------------------------------------
# Include directories, compiler flags, libraries

include_directories(${CH_IRRLICHT_INCLUDES})
set(COMPILER_FLAGS "${CH_CXX_FLAGS}  ${CH_IRRLICHT_CXX_FLAGS}")
set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")
list(APPEND LIBS "")

if(ENABLE_MODULE_PARDISO_MKL)
    include_directories(${CH_MKL_INCLUDES})
    set(COMPILER_FLAGS "${COMPILER_FLAGS} ${CH_MKL_CXX_FLAGS}")
    set(LINKER_FLAGS "${LINKER_FLAGS} ${CH_MKL_LINK_FLAGS}")
    list(APPEND LIBS "ChronoEngine_pardisomkl")
endif()

if(ENABLE_MODULE_MUMPS)
    include_directories(${CH_MUMPS_INCLUDES})
    list(APPEND LIBS "ChronoEngine_mumps")
endif()

#--------------------------------------------------------------
# Create the executables

foreach(PROGRAM ${DEMOS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM} ${PROGRAM}.cpp)
    source_group("" FILES ${PROGRAM}.cpp)

    set_target_properties(${PROGRAM} PROPERTIES
                          COMPILE_FLAGS "${COMPILER_FLAGS}"
                          LINK_FLAGS "${LINKER_FLAGS}")
    if(APPLE)
      set_target_properties(${PROGRAM} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${MACOSX_BUNDLE_INFO_PLIST})
    endif()
    set_property(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
    target_link_libraries(${PROGRAM}
                          ChronoEngine
                          ChronoEngine_irrlicht
                          ChronoEngine_vehicle
                          ChronoEngine_vehicle_irrlicht
                          ChronoModels_vehicle
                          ${LIBS})
endforeach(PROGRAM)
