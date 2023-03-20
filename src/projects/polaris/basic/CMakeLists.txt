#=============================================================================
# CMake configuration file for various Polaris tests
#=============================================================================

if(NOT ENABLE_MODULE_IRRLICHT)
    return()
endif()

#--------------------------------------------------------------
# List all programs

set(PROGRAMS
    test_Polaris
    test_Polaris_Accel
    test_Polaris_DLC
    test_Polaris_Turn
    test_Polaris_CustomTerrain
)

#--------------------------------------------------------------
# Include directories, compiler flags, libraries

include_directories(${CH_IRRLICHT_INCLUDES})
set(COMPILER_FLAGS "${CH_CXX_FLAGS}  ${CH_IRRLICHT_CXX_FLAGS}")
set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")
list(APPEND LIBS "")

#--------------------------------------------------------------
# Create the executables

foreach(PROGRAM ${PROGRAMS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM} ${PROGRAM}.cpp)
    source_group("" FILES ${PROGRAM}.cpp)

    set_target_properties(${PROGRAM} PROPERTIES
                          COMPILE_FLAGS "${COMPILER_FLAGS}"
                          LINK_FLAGS "${LINKER_FLAGS}")
    if(APPLE)
      set_target_properties(${PROGRAM} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${MACOSX_BUNDLE_INFO_PLIST})
    endif()
    set_property(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
    target_link_libraries(${PROGRAM}
                          ChronoEngine
                          ChronoEngine_irrlicht
                          ChronoEngine_vehicle
                          ChronoEngine_vehicle_irrlicht
                          ChronoModels_vehicle
                          ${LIBS})
endforeach()
