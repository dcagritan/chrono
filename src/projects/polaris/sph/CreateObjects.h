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
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"

chrono::ChCoordsys<> CreateTerrain(chrono::ChSystem& sys,
                                   chrono::fsi::ChSystemFsi& sysFSI,
                                   const std::string& terrain_dir,
                                   double ramp_length,
                                   bool terrain_mesh_vis,
                                   bool terrain_mesh_contact);

std::shared_ptr<chrono::ChBezierCurve> CreatePath(const std::string& terrain_dir, double ramp_length);
std::shared_ptr<chrono::ChBody> CreateSentinel(chrono::ChSystem& sys, const chrono::ChCoordsys<>& init_pos);

std::shared_ptr<chrono::vehicle::WheeledVehicle> CreateVehicle(chrono::ChSystem& sys,
                                                               const chrono::ChCoordsys<>& init_pos);

void CreateWheelBCEMarkers(std::shared_ptr<chrono::vehicle::WheeledVehicle> vehicle, chrono::fsi::ChSystemFsi& sysFSI);
