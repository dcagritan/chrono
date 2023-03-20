if(NOT ENABLE_MODULE_IRRLICHT)
  return()
endif()  

# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${CH_IRRLICHT_INCLUDES})

SET(LIBRARIES
    ChronoEngine
    ChronoEngine_irrlicht
    ChronoEngine_vehicle
    ChronoEngine_vehicle_irrlicht
    ChronoModels_vehicle
)

# ------------------------------------------------------------------------------
# List of all executables
# ------------------------------------------------------------------------------

SET(DEMOS
    test_WVP_sequential
    test_WVP_suspensionTestRig
    test_WVP_straight_slow
    test_WVP_constant_radius
    test_WVP_gravel_lane_change
    test_WVP_paved_lane_change
    test_WVP_side_slope
    test_WVP_straight_accel
    test_WVP_wall2wall
    test_WVP_pavedDLC_closed
    test_WVP_gravelDLC_closed
    test_WVP_slope_grade
    test_WVP_tractive_effort
    test_WVP_ride_quality
    test_WVP_softsoil_traction
    test_WVP_drawbar_pull
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

FOREACH(PROGRAM ${DEMOS})
    MESSAGE(STATUS "...add ${PROGRAM}")

    ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
        FOLDER demos
        COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}"
    )

    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})

ENDFOREACH(PROGRAM)
