#=============================================================================
# CMake configuration file for the multicore tracked vehicle tests.
#=============================================================================

#--------------------------------------------------------------
# Check dependencies.

IF(NOT ENABLE_MODULE_OPENGL OR NOT ENABLE_MODULE_IRRLICHT OR NOT ENABLE_MODULE_MULTICORE)
  RETURN()
ENDIF()

#--------------------------------------------------------------
# List all model files for this demo

SET(PROGRAMS 
  test_VEH_M113_seqA
  test_VEH_M113_mcoreA
  test_VEH_M113_mcoreB
  test_VEH_Marder_mcoreA
)

#--------------------------------------------------------------
# Additional include directories

INCLUDE_DIRECTORIES(${CH_MULTICORE_INCLUDES})
INCLUDE_DIRECTORIES(${CH_OPENGL_INCLUDES})
INCLUDE_DIRECTORIES(${CH_IRRLICHT_INCLUDES})

#--------------------------------------------------------------
# Add executables

FOREACH(PROGRAM ${PROGRAMS})
    MESSAGE(STATUS "...add ${PROGRAM}")

    ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
        FOLDER demos
        COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_MULTICORE_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}"
    )
    SET_PROPERTY(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
    TARGET_LINK_LIBRARIES(${PROGRAM} 
                      ChronoEngine
                      ChronoEngine_vehicle
                      ChronoEngine_multicore
                      ChronoEngine_irrlicht
                      ChronoEngine_vehicle_irrlicht
                      ChronoEngine_opengl
                      ChronoModels_vehicle)
ENDFOREACH(PROGRAM)
