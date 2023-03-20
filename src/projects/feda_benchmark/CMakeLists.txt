message(STATUS "Projects: FED-alpha benchmark...")

if(NOT ENABLE_MODULE_VEHICLE OR NOT ENABLE_MODULE_IRRLICHT)
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
    test_FEDA_sequential
    test_FEDA_wall2wall_turning
    test_FEDA_accel
    test_FEDA_lanechange
    test_FEDA_steady_state_cornering
)

if(HAVE_OPENCRG)
   set(DEMOS ${DEMOS}
       test_FEDA_shock
       test_FEDA_ride
   )
endif()

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
