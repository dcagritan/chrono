// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Alessandro Tasora, Radu Serban
// =============================================================================

#include "chrono/solver/ChSolver.h"

namespace chrono {

// Trick to avoid putting the following mapper macro inside the class definition in .h file:
// enclose macros in local 'my_enum_mappers', just to avoid avoiding cluttering of the parent class.
class my_enum_mappers : public ChSolver {
  public:
    CH_ENUM_MAPPER_BEGIN(Type);
    CH_ENUM_VAL(SOR);
    CH_ENUM_VAL(SYMMSOR);
    CH_ENUM_VAL(JACOBI);
    CH_ENUM_VAL(SOR_MULTITHREAD);
    CH_ENUM_VAL(PMINRES);
    CH_ENUM_VAL(BARZILAIBORWEIN);
    CH_ENUM_VAL(PCG);
    CH_ENUM_VAL(APGD);
    CH_ENUM_VAL(MINRES);
    CH_ENUM_VAL(SOLVER_DEM);
    CH_ENUM_VAL(CUSTOM);
    CH_ENUM_MAPPER_END(Type);
};

void ChSolver::ArchiveOUT(ChArchiveOut& marchive) {
    // version number
    marchive.VersionWrite(1);
    // solver type:
    my_enum_mappers::Type_mapper typemapper;
    Type type = GetType();
    marchive << CHNVP(typemapper(type), "solver_type");
    // serialize all member data:
    marchive << CHNVP(verbose);
}

void ChSolver::ArchiveIN(ChArchiveIn& marchive) {
    // version number
    int version = marchive.VersionRead();
    // solver type:
    my_enum_mappers::Type_mapper typemapper;
    Type type = GetType();
    marchive >> CHNVP(typemapper(type), "solver_type");
    // stream in all member data:
    marchive >> CHNVP(verbose);
}

}  // end namespace chrono
