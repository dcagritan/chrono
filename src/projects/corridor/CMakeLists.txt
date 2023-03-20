# Test programs for AV corridor simulation
# ------------------------------------------------------------------------------

message(STATUS "Projects: autonomous vehicles...")

if(NOT ENABLE_MODULE_VEHICLE)
  return()
endif()

if(NOT ENABLE_MODULE_IRRLICHT)
  return()
endif()

# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

include_directories(${CH_IRRLICHT_INCLUDES})

# ------------------------------------------------------------------------------

set(PROGRAMS
    test_AV_corridor
)

set(MODEL_FILES
    framework.h
    framework.cpp
    agent.h
    agent.cpp
    message.h
    message.cpp
    scene.h
    scene.cpp
    irrapp.h
    irrapp.cpp
    path.h
    path.cpp
    traffic_light.h
    traffic_light.cpp
    vehicle.h
    vehicle.cpp
    truckAV.h
    truckAV.cpp
    vanAV.h
    vanAV.cpp
    sedanAV.h
    sedanAV.cpp
    ChSensor.h
    ChSensor.cpp
    ChCollisionLidar.h
    ChCollisionLidar.cpp
    ChCollisionLidarRay.h
    ChCollisionLidarRay.cpp
    ChMAPMessage.h
    ChMAPMessage.cpp
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

foreach(PROGRAM ${PROGRAMS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp" ${MODEL_FILES})
    source_group(""  FILES "${PROGRAM}.cpp" ${MODEL_FILES})

    set_target_properties(${PROGRAM} PROPERTIES
        FOLDER tests
        COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}"
    )

    target_link_libraries(${PROGRAM}
                          ChronoEngine
                          ChronoEngine_irrlicht
                          ChronoEngine_vehicle
                          ChronoEngine_vehicle_irrlicht
                          ChronoModels_vehicle)

endforeach(PROGRAM)

# ------------------------------------------------------------------------------

set(PROGRAM test_AV_fsm)
message(STATUS "...add ${PROGRAM}")
add_executable(${PROGRAM}  "${PROGRAM}.cpp")
source_group(""  FILES "${PROGRAM}.cpp")
set_target_properties(${PROGRAM} PROPERTIES
    FOLDER tests
    COMPILE_FLAGS "${CH_CXX_FLAGS}"
    LINK_FLAGS "${CH_LINKERFLAG_EXE}"
)
