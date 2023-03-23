if(NOT ENABLE_MODULE_VEHICLE)
    return()
endif()

message(STATUS "Projects: Polaris...")

add_subdirectory(basic)
add_subdirectory(sph)
add_subdirectory(sph_nn)
add_subdirectory(scm)
