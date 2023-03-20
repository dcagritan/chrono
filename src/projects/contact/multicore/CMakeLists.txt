if(NOT ENABLE_MODULE_MULTICORE)
  return()
endif()

# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${CH_MULTICORE_INCLUDES})

# Add programs that require OpenGL
IF(ENABLE_MODULE_OPENGL)
    INCLUDE_DIRECTORIES(${CH_OPENGL_INCLUDES})

    IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       SET (CH_LINKERFLAG_EXE  "${CH_LINKERFLAG_EXE} -framework IOKit -framework Cocoa -framework OpenGL")
    ENDIF()

    set(PROGRAMS_GL
        test_CTC_box_collision_mcore
        test_CTC_rolling_mcore
        test_CTC_mesh_mcore
    )
ENDIF()

# Add programs that require Irrlicht
if(ENABLE_MODULE_IRRLICHT)
	include_directories(${CH_IRRLICHT_INCLUDES})

	set(PROGRAMS_IRR
		test_CTC_sliding_gravity_mcore
    )
endif()


FOREACH(PROGRAM ${PROGRAMS_GL})
    MESSAGE(STATUS "...add ${PROGRAM}")

    ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
        FOLDER demos
        COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_MULTICORE_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}")
    SET_PROPERTY(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
    TARGET_LINK_LIBRARIES(${PROGRAM} ChronoEngine ChronoEngine_multicore ChronoEngine_opengl)

ENDFOREACH(PROGRAM)

FOREACH(PROGRAM ${PROGRAMS_IRR})
    MESSAGE(STATUS "...add ${PROGRAM}")

    ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
    SOURCE_GROUP(""  FILES  "${PROGRAM}.cpp")

    SET_TARGET_PROPERTIES(${PROGRAM}  PROPERTIES
        FOLDER demos
        COMPILE_FLAGS "${CH_CXX_FLAGS}  ${CH_MULTICORE_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}")
    SET_PROPERTY(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
    TARGET_LINK_LIBRARIES(${PROGRAM} ChronoEngine ChronoEngine_multicore ChronoEngine_irrlicht)
ENDFOREACH()

