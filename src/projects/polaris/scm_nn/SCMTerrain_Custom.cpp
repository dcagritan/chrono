// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Alessandro Tasora, Radu Serban, Jay Taves
// =============================================================================
//
// Deformable terrain based on SCM (Soil Contact Model) from DLR
// (Krenn & Hirzinger)
//
// =============================================================================

#include <cstdio>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <limits>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "chrono/physics/ChMaterialSurfaceNSC.h"
#include "chrono/physics/ChMaterialSurfaceSMC.h"
#include "chrono/assets/ChTexture.h"
#include "chrono/assets/ChBoxShape.h"
#include "chrono/utils/ChConvexHull.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "SCMTerrain_Custom.h"

#include "chrono_thirdparty/stb/stb.h"

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// Implementation of the SCMTerrain wrapper class
// -----------------------------------------------------------------------------

SCMTerrain_Custom::SCMTerrain_Custom(ChSystem* system, bool visualization_mesh) :SCMTerrain(system,visualization_mesh) {
}


}  // end namespace vehicle
}  // end namespace chrono
