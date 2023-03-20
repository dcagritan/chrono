if(NOT ENABLE_MODULE_IRRLICHT)
  return()
endif()  

# ------------------------------------------------------------------------------

set(DEMOS
    test_BASE_convex_hull
    test_BASE_motors
    test_BASE_filesystem
    test_BASE_body_force_seq
    test_BASE_collision_family
    test_BASE_rev_motion
    test_BASE_equilibrium
    test_BASE_archive
    test_BASE_lock_lock
)

include_directories(${CH_IRRLICHT_INCLUDES})

foreach(PROGRAM ${DEMOS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp")
    source_group(""  FILES "${PROGRAM}.cpp")

    set_target_properties(${PROGRAM} PROPERTIES
        FOLDER tests
        COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}"
    )
    target_link_libraries(${PROGRAM} ChronoEngine ChronoEngine_irrlicht)
endforeach(PROGRAM)
