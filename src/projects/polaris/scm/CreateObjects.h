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
// Author: Radu Serban
// =============================================================================
//
// Generator functions for Polaris on SPH terrain system
//
// =============================================================================

#pragma once

#include "chrono/physics/ChSystem.h"
#include "chrono/core/ChBezierCurve.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"


std::shared_ptr<chrono::vehicle::WheeledVehicle> CreateVehicle(chrono::ChSystem& sys,
                                                               const chrono::ChCoordsys<>& init_pos);