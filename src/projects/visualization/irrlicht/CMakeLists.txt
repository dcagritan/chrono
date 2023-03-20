if(NOT ENABLE_MODULE_IRRLICHT)
  return()
endif()  

# ------------------------------------------------------------------------------

set(PROGRAMS_SEQ
    test_VIS_line
    test_VIS_color_mesh
)

include_directories(${CH_IRRLICHT_INCLUDES})

foreach(PROGRAM ${PROGRAMS_SEQ})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp")
    source_group(""  FILES "${PROGRAM}.cpp")

    set_target_properties(${PROGRAM} PROPERTIES
        FOLDER demos
        COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}"
    )

    target_link_libraries(${PROGRAM} ChronoEngine ChronoEngine_irrlicht)

endforeach(PROGRAM)

# ------------------------------------------------------------------------------

if(NOT ENABLE_MODULE_MULTICORE)
  return()
endif()

set(PROGRAMS_PAR
   test_VIS_plate_balls
)

include_directories(${CH_MULTICORE_INCLUDES})

foreach(PROGRAM ${PROGRAMS_PAR})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp")
    source_group(""  FILES "${PROGRAM}.cpp")

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
