#=============================================================================
# M113 sequential test and validation programs
#=============================================================================

# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

set(LIBRARIES
    ChronoEngine
    ChronoEngine_vehicle
    ChronoModels_vehicle
)

set(CXX_FLAGS "${CH_CXX_FLAGS}")

if(ENABLE_MODULE_IRRLICHT)
  set(LIBRARIES ${LIBRARIES} ChronoEngine_irrlicht ChronoEngine_vehicle_irrlicht)
  include_directories(${CH_IRRLICHT_INCLUDES})
  set(CXX_FLAGS "${CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}")
endif()

#--------------------------------------------------------------
# List of all executables
#--------------------------------------------------------------

set(TERRAIN_FILES
  ../terrain/RigidTerrainSlope.h
  ../terrain/RigidTerrainSlope.cpp
  ../terrain/RigidTerrainStep.h
  ../terrain/RigidTerrainStep.cpp
  ../terrain/RigidTerrainTrapezoid.h
  ../terrain/RigidTerrainTrapezoid.cpp
)

set(DEMOS
  test_M113a
  test_M113a_acceleration
  test_M113a_turnInPlace
  test_M113a_steadyStateCornering
  test_M113a_doubleLaneChange
  test_M113a_sideSlopeStability
  test_M113a_sideSlopeStability_defSoil
  test_M113a_roughTerrain
  test_M113a_halfRoundObstacle
  test_M113a_stepClimb
  test_M113a_trapezoidGapBarrierDitch
  test_M113a_offRoadTrafficability
  test_M113a_gradeClimbing_defSoil
  test_M113a_speedOnGrade
  test_M113a_fuelEconomy
)

#--------------------------------------------------------------
# Add all executables
#--------------------------------------------------------------

foreach(PROGRAM ${DEMOS})
  message(STATUS "...add ${PROGRAM}")

  add_executable(${PROGRAM} ${PROGRAM}.cpp ${TERRAIN_FILES})
  source_group("terrain" FILES ${TERRAIN_FILES})
  source_group(""  FILES ${PROGRAM}.cpp ${TERRAIN_FILES})

  set_target_properties(${PROGRAM} PROPERTIES
    FOLDER tests
    COMPILE_FLAGS "${CXX_FLAGS}"
    LINK_FLAGS "${CH_LINKERFLAG_EXE}"
  )

  target_link_libraries(${PROGRAM} ${LIBRARIES})

endforeach(PROGRAM)
