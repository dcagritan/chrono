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
// Authors: Radu Serban
// =============================================================================
//
// Main driver function for a Polaris MRZR specified through JSON files.
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#include "chrono/core/ChRealtimeStep.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/driver/ChIrrGuiDriver.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"

#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledTrailer.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::vehicle;

// =============================================================================

enum class MRZR_MODEL { ORIGINAL, MODIFIED };

MRZR_MODEL model = MRZR_MODEL::MODIFIED;

// -----------------------------------------------------------------------------

// JSON files for terrain
std::string rigidterrain_file("terrain/RigidPlane.json");
////std::string rigidterrain_file("terrain/RigidMesh.json");
////std::string rigidterrain_file("terrain/RigidHeightMap.json");
////std::string rigidterrain_file("terrain/RigidSlope10.json");
////std::string rigidterrain_file("terrain/RigidSlope20.json");

// Initial vehicle position and orientation
ChVector<> initLoc(0, 0, 0.5);
double initYaw = 20 * CH_C_DEG_TO_RAD;

// Simulation step size
double step_size = 2e-3;

// =============================================================================

int main(int argc, char* argv[]) {
    std::string model_dir = (model == MRZR_MODEL::ORIGINAL) ? "mrzr/JSON_orig/" : "mrzr/JSON_new/";

    std::string vehicle_json = model_dir + "vehicle/MRZR.json";
    ////std::string powertrain_json = model_dir + "powertrain/MRZR_SimplePowertrain.json";
    std::string powertrain_json = model_dir + "powertrain/MRZR_SimpleMapPowertrain.json";
    std::string tire_json = model_dir + "tire/MRZR_TMeasyTire.json";

    // Create the vehicle system
    WheeledVehicle vehicle(vehicle::GetDataFile(vehicle_json), ChContactMethod::SMC);
    vehicle.Initialize(ChCoordsys<>(initLoc, Q_from_AngZ(initYaw)));
    vehicle.GetChassis()->SetFixed(false);
    vehicle.SetChassisVisualizationType(VisualizationType::NONE);
    vehicle.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(powertrain_json));
    vehicle.InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : vehicle.GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(tire_json));
            vehicle.InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    // Containing system
    auto system = vehicle.GetSystem();

    // Create the terrain
    RigidTerrain terrain(system, vehicle::GetDataFile(rigidterrain_file));
    terrain.Initialize();

    // Create Irrilicht visualization
    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("Polaris MRZR - JSON specification");
    vis->SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), 5.0, 0.5);
    vis->Initialize();
    vis->AddTypicalLights();
    vis->AddSkyBox();
    vis->AddLogo();
    vehicle.SetVisualSystem(vis);

    // Create the interactive driver
    ChIrrGuiDriver driver(*vis);
    driver.SetSteeringDelta(0.02);
    driver.SetThrottleDelta(0.02);
    driver.SetBrakingDelta(0.06);
    driver.Initialize();

    // Modify solver settings if the vehicle model contains bushings
    if (vehicle.HasBushings()) {
        auto solver = chrono_types::make_shared<ChSolverMINRES>();
        system->SetSolver(solver);
        solver->SetMaxIterations(150);
        solver->SetTolerance(1e-10);
        solver->EnableDiagonalPreconditioner(true);
        solver->EnableWarmStart(true);
        solver->SetVerbose(false);

        step_size = 2e-4;
        system->SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);
    }

    // Simulation loop
    ChRealtimeStepTimer realtime_timer;
    while (vis->Run()) {
        // Render scene
        vis->BeginScene();
        vis->DrawAll();
        vis->EndScene();

        // Get driver inputs
        ChDriver::Inputs driver_inputs = driver.GetInputs();

        // Update modules (process inputs from other modules)
        double time = vehicle.GetSystem()->GetChTime();
        driver.Synchronize(time);
        vehicle.Synchronize(time, driver_inputs, terrain);
        terrain.Synchronize(time);
        vis->Synchronize("Polaris MRZR", driver_inputs);

        // Advance simulation for one timestep for all modules
        driver.Advance(step_size);
        vehicle.Advance(step_size);
        terrain.Advance(step_size);
        vis->Advance(step_size);

        // Spin in place for real time to catch up
        realtime_timer.Spin(step_size);
    }

    return 0;
}