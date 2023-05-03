// // =============================================================================
// // PROJECT CHRONO - http://projectchrono.org
// //
// // Copyright (c) 2022 projectchrono.org
// // All rights reserved.
// //
// // Use of this source code is governed by a BSD-style license that can be found
// // in the LICENSE file at the top level of the distribution and at
// // http://projectchrono.org/license-chrono.txt.
// //
// // =============================================================================
// // Author: Radu Serban
// // =============================================================================
// //
// // Generator functions for Polaris on SPH terrain system
// //
// // =============================================================================

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "chrono/assets/ChBoxShape.h"
#include "chrono/assets/ChSphereShape.h"
#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "CreateObjects.h"

using namespace chrono;
using namespace chrono::vehicle;


std::shared_ptr<WheeledVehicle> CreateVehicle(ChSystem& sys,
                                              const ChCoordsys<>& init_pos) {
    std::string vehicle_json = "Polaris/Polaris.json";
    std::string powertrain_json = "Polaris/Polaris_SimpleMapPowertrain.json";
    std::string tire_json = "Polaris/Polaris_RigidTire.json";

    // Create and initialize the vehicle
    auto vehicle = chrono_types::make_shared<WheeledVehicle>(&sys, vehicle::GetDataFile(vehicle_json));
    vehicle->Initialize(init_pos);
    vehicle->GetChassis()->SetFixed(false);
    vehicle->SetChassisVisualizationType(VisualizationType::MESH);
    vehicle->SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle->SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    vehicle->SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(powertrain_json));
    vehicle->InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : vehicle->GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(tire_json));
            vehicle->InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    return vehicle;
}

