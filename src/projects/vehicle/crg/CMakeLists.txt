if(NOT ENABLE_MODULE_IRRLICHT OR NOT HAVE_OPENCRG)
    return()
endif()

#--------------------------------------------------------------
# List all demos

set(DEMOS
    test_VEH_rms
    )

#--------------------------------------------------------------
# Additional include directories

include_directories(${CH_IRRLICHT_INCLUDES})

#--------------------------------------------------------------
# Create the executables

foreach(PROGRAM ${DEMOS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM} ${PROGRAM}.cpp)
    source_group("" FILES ${PROGRAM}.cpp)
    set_target_properties(${PROGRAM} PROPERTIES 
                          COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
                          LINK_FLAGS "${CH_LINKERFLAG_EXE}")
    target_link_libraries(${PROGRAM}
                          ChronoEngine
                          ChronoEngine_irrlicht
                          ChronoEngine_vehicle
                          ChronoEngine_vehicle_irrlicht
                          ChronoModels_vehicle)
endforeach()
