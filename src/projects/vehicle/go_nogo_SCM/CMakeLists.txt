# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

set(LIBRARIES
    ChronoEngine
    ChronoEngine_vehicle
    ChronoModels_vehicle
)

if (ENABLE_MODULE_IRRLICHT)
    include_directories(${CH_IRRLICHT_INCLUDES})
    set(CXX_FLAGS "${CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}")
    set(LIBRARIES ${LIBRARIES} ChronoEngine_irrlicht ChronoEngine_vehicle_irrlicht)
endif()

# ------------------------------------------------------------------------------
# List of all executables
# ------------------------------------------------------------------------------

set(DEMOS
    test_VEH_gonogo_m113
    test_VEH_halfround_m113
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

foreach(PROGRAM ${DEMOS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  ${PROGRAM}.cpp)
    source_group(""  FILES ${PROGRAM}.cpp)
    set_target_properties(${PROGRAM} PROPERTIES
        FOLDER tests
        COMPILE_FLAGS "${CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}")
    target_link_libraries(${PROGRAM} ${LIBRARIES})
endforeach(PROGRAM)
