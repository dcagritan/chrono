message(STATUS "Projects: FEA module...")

if(NOT ENABLE_MODULE_IRRLICHT OR NOT ENABLE_MODULE_PARDISO_MKL)
  return()
endif()  

#--------------------------------------------------------------

set(FEA_DEMOS_IRR_MKL
    test_FEA_ancf_cable
    test_FEA_ancf_cable2
    test_FEA_constraints
    test_FEA_ancf_shell
    test_FEA_ancf_beam3_Princeton
)

include_directories(CMAKE_CURRENT_SOURCE_DIR)
include_directories(${CH_IRRLICHTINC})
include_directories(${CH_MKL_INCLUDES})
list(APPEND LIBS "ChronoEngine")
list(APPEND LIBS "ChronoEngine_irrlicht")
list(APPEND LIBS "ChronoEngine_pardisomkl")
set(COMPILER_FLAGS "${CH_CXX_FLAGS} ${CH_MKL_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}")
set(LINKER_FLAGS "${CH_LINKERFLAG_EXE} ${CH_MKL_LINK_FLAGS}")

if(ENABLE_MODULE_MUMPS)
  include_directories(${CH_MUMPS_INCLUDES})
  list(APPEND LIBS "ChronoEngine_mumps")
endif()

#--------------------------------------------------------------

foreach(PROGRAM ${FEA_DEMOS_IRR_MKL})
    MESSAGE(STATUS "...add ${PROGRAM}")

    ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
    SOURCE_GROUP(""  FILES  "${PROGRAM}.cpp")   

    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES 
        FOLDER demos
        COMPILE_FLAGS "${COMPILER_FLAGS}"
        LINK_FLAGS "${LINKER_FLAGS}")
        
    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBS})

endforeach()






