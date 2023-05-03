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
#ifndef SCM_TERRAIN_CUSTOM_H
#define SCM_TERRAIN_CUSTOM_H

#include <string>
#include <ostream>
#include <unordered_map>

#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChLoadsBody.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/core/ChTimer.h"

#include "chrono_vehicle/ChApiVehicle.h"
#include "chrono_vehicle/ChSubsysDefs.h"
#include "chrono_vehicle/ChTerrain.h"
#include "chrono_vehicle/ChWorldFrame.h"
#include "chrono_vehicle/terrain/SCMTerrain.h"

namespace chrono {
namespace vehicle {


/// @addtogroup vehicle_terrain
/// @{

/// Deformable terrain model.
/// This class implements a deformable terrain based on the Soil Contact Model.
/// Unlike RigidTerrain, the vertical coordinates of this terrain mesh can be deformed
/// due to interaction with ground vehicles or other collision shapes.
class CH_VEHICLE_API SCMTerrain_Custom : public SCMTerrain {
  public:

    /// Construct a default SCM deformable terrain.
    /// The user is responsible for calling various Set methods before Initialize.
    SCMTerrain_Custom(ChSystem* system,               ///< [in] containing multibody system
               bool visualization_mesh = true  ///< [in] enable/disable visualization asset
    );

    ~SCMTerrain_Custom() {}

};


/// @} vehicle_terrain

}  // end namespace vehicle
}  // end namespace chrono
#endif