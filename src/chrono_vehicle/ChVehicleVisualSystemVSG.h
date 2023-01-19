// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2022 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Rainer Gericke, Radu Serban
// =============================================================================
//
// VSG-based visualization wrapper for vehicles.  This class is a derived
// from ChVisualSystemVSG and provides the following functionality:
//   - rendering of the entire Irrlicht scene
//   - custom chase-camera (which can be controlled with keyboard)
//   - optional rendering of links, springs, stats, etc.
//
// =============================================================================

#ifndef CH_VEHICLE_VISUAL_SYSTEM_VSG_H
#define CH_VEHICLE_VISUAL_SYSTEM_VSG_H

#include <string>

#include "chrono/physics/ChSystem.h"
#include "chrono/utils/ChUtilsChaseCamera.h"

#include "chrono_vsg/ChVisualSystemVSG.h"

#include "chrono_vehicle/ChApiVehicle.h"
#include "chrono_vehicle/ChVehicle.h"
#include "chrono_vehicle/ChVehicleVisualSystem.h"
#include "chrono_vehicle/ChDriver.h"
#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"

namespace chrono {
namespace vehicle {

class ChVSGGuiDriver;

/// @addtogroup vehicle
/// @{

/// VSG-based Chrono run-time visualization system.
class CH_VEHICLE_API ChVehicleVisualSystemVSG : public ChVehicleVisualSystem, public vsg3d::ChVisualSystemVSG {
  public:
    /// Construct a vehicle VSG visualization system
    ChVehicleVisualSystemVSG();

    virtual ~ChVehicleVisualSystemVSG();

    /// Initialize the visualization system.
    virtual void Initialize() override;

    /// Advance the dynamics of the chase camera.
    /// The integration of the underlying ODEs is performed using as many steps as needed to advance
    /// by the specified duration.
    virtual void Advance(double step) override;

    void SetTargetSymbol(double size, ChColor col);
    void SetTargetSymbolPosition(ChVector<> pos);
    void SetSentinelSymbol(double size, ChColor col);
    void SetSentinelSymbolPosition(ChVector<> pos);

  protected:
    virtual void AppendGUIStats() {}

    ChVSGGuiDriver* m_guiDriver;
    bool m_has_TC;

    vsg::dvec3 m_target_symbol_position = vsg::dvec3(0.0, 0.0, 0.0);
    vsg::dvec3 m_target_symbol_size = vsg::dvec3(1.0, 1.0, 1.0);
    vsg::dvec3 m_sentinel_symbol_position = vsg::dvec3(0.0, 0.0, 0.0);
    vsg::dvec3 m_sentinel_symbol_size = vsg::dvec3(1.0, 1.0, 1.0);

    friend class ChVSGGuiDriver;
    friend class ChVehicleGuiComponentVSG;
    friend class ChVehicleKeyboardHandlerVSG;
};

// @} vehicle

}  // namespace vehicle
}  // namespace chrono
#endif