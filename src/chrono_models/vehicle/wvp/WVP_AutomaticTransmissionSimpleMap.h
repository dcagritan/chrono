// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2023 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban, Asher Elmquist, Marcel Offermans
// =============================================================================
//
// Simple automatic transmission model for the Sedan vehicle.
// - no torque converter
// - simple gear-shifting model (in automatic mode)
//
// =============================================================================

#ifndef WVP_AUTOMATICTRANSMISSIONSIMPLEMAP_H
#define WVP_AUTOMATICTRANSMISSIONSIMPLEMAP_H

#include "chrono_vehicle/powertrain/ChAutomaticTransmissionSimpleMap.h"
#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace wvp {

/// @addtogroup vehicle_models_sedan
/// @{

/// Simple Sedan automatic transmission subsystem.
class CH_MODELS_API WVP_AutomaticTransmissionSimpleMap : public ChAutomaticTransmissionSimpleMap {
  public:
    WVP_AutomaticTransmissionSimpleMap(const std::string& name);

    /// Set the transmission gear ratios (one or more forward gear ratios and a single reverse gear ratio).
    void SetGearRatios(std::vector<double>& fwd, double& rev) override;

    /// Set the ideal shift points for automatic gear shifting.
    /// For each forward gear, specify a pair (min, max) with the minimum and
    /// maximum engine speed for shifting (down and up, respectively).
    void SetShiftPoints(
        std::vector<std::pair<double, double>>& shift_bands  ///< [out] down-shift/up-shift points
        ) override;
};

/// @} vehicle_models_sedan

}  // end namespace sedan
}  // end namespace vehicle
}  // end namespace chrono

#endif
