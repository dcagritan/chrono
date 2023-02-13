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
// Authors: Radu Serban
// =============================================================================
//
// WVP suspension test rig.
//
// Driver inputs for a suspension test rig include left/right post displacements
// and steering input (the latter being ignored if the tested suspension is not
// attached to a steering mechanism).  These driver inputs can be obtained from
// an interactive driver system (of type ChIrrGuiDriverSTR) or from a data file
// (using a driver system of type ChDataDriverSTR).
//
// If data collection is enabled, an output file named 'output.dat' will be
// generated in the directory specified by the variable out_dir. This ASCII file
// contains one line per output time, each with the following information:
//  [col  1]     time
//  [col  2]     left post input, a value in [-1,1]
//  [col  3]     right post input, a value in [-1,1]
//  [col  4]     steering input, a value in [-1,1]
//  [col  5]     actual left post dispalcement
//  [col  6]     actual right post displacement
//  [col  7- 9]  application point for left tire force
//  [col 10-12]  left tire force
//  [col 13-15]  left tire moment
//  [col 16-18]  application point for right tire force
//  [col 19-21]  right tire force
//  [col 22-24]  right tire moment
//
// Tire forces are expressed in the global frame, as applied to the center of
// the associated wheel.
//
// =============================================================================

#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/wheeled_vehicle/test_rig/ChSuspensionTestRig.h"
#include "chrono_vehicle/wheeled_vehicle/test_rig/ChSuspensionTestRigInteractiveDriverIRR.h"
#include "chrono_vehicle/wheeled_vehicle/test_rig/ChSuspensionTestRigDataDriver.h"
#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_models/vehicle/wvp/WVP_Vehicle.h"
#include "chrono_models/vehicle/wvp/WVP_RigidTire.h"
#include "chrono_models/vehicle/wvp/WVP_TMeasyTire.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::wvp;

// =============================================================================
// USER SETTINGS
// =============================================================================

// Simulation step size
double step_size = 1e-3;

// Axle index
int axle_index = 0;
double post_limit = 0.15;

// Specification of test rig inputs:
//   'true':  use driver inputs from file
//   'false': use interactive Irrlicht driver
bool use_data_driver = false;

// File with driver inputs
std::string driver_file("hmmwv/suspensionTest/ST_inputs.dat");

// Output collection
bool collect_output = false;
std::string out_dir = "../WVP_TEST_RIG";
double out_step_size = 1.0 / 100;

// =============================================================================
int main(int argc, char* argv[]) {
    // Create and initialize a WVP vehicle.
    auto vehicle = chrono_types::make_shared<WVP_Vehicle>(true);

    // Create and intialize the suspension test rig.
    int steering_index = (axle_index == 0) ? 0 : -1;
    ChSuspensionTestRigPlatform rig(vehicle, {axle_index}, post_limit);
    if (axle_index == 0)
        rig.IncludeSteeringMechanism(steering_index);

    // Create the tires
    for (auto& axle : vehicle->GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = chrono_types::make_shared<WVP_TMeasyTire>("");
            vehicle->InitializeTire(tire, wheel, VisualizationType::NONE);
        }
    }

    rig.SetInitialRideHeight(0.5);

    rig.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    rig.SetWheelVisualizationType(VisualizationType::PRIMITIVES);
    rig.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    rig.SetTireVisualizationType(VisualizationType::PRIMITIVES);

    // Create the vehicle Irrlicht application.
    auto vis = chrono_types::make_shared<ChVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("WVP Suspension Test Rig");
    vis->SetChaseCamera(0.5 * (rig.GetSpindlePos(0, LEFT) + rig.GetSpindlePos(0, RIGHT)), 2.0, 1.0);

    // Create and initialize the driver system.
    if (use_data_driver) {
        // Driver with inputs from file
        auto driver = chrono_types::make_shared<ChSuspensionTestRigDataDriver>(vehicle::GetDataFile(driver_file));
        rig.SetDriver(driver);
    } else {
        // Interactive driver
        auto driver = chrono_types::make_shared<ChSuspensionTestRigInteractiveDriverIRR>(*vis);
        driver->SetSteeringDelta(1.0 / 50);
        driver->SetDisplacementDelta(1.0 / 250);
        rig.SetDriver(driver);
    }

    // Initialize suspension test rig.
    rig.Initialize();

    vis->Initialize();
    vis->AddTypicalLights();
    vis->AddSkyBox();
    vis->AddLogo();
    vis->AttachVehicle(&rig.GetVehicle());

    // Initialize output
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cout << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    std::string out_file = out_dir + "/output.dat";
    utils::CSV_writer out_csv(" ");

    // ---------------
    // Simulation loop
    // ---------------

    // Number of simulation steps between two data collection frames
    int out_steps = (int)std::ceil(out_step_size / step_size);

    // Initialize simulation frame counter
    int step_number = 0;

    while (vis->Run()) {
        // Render scene
        vis->BeginScene();
        vis->Render();
        vis->EndScene();

        // Write output data
        if (collect_output && step_number % out_steps == 0) {
            // Current tire forces
            out_csv << rig.GetActuatorDisp(0, LEFT) << rig.GetActuatorDisp(0, RIGHT);
            out_csv << std::endl;
        }

        // Advance simulation of the rig
        rig.Advance(step_size);

        // Update visualization app
        vis->Synchronize(rig.GetVehicle().GetChTime(), {rig.GetSteeringInput(), 0, 0});
        vis->Advance(step_size);

        // Increment frame number
        step_number++;
    }

    // Write output file
    if (collect_output) {
        out_csv.write_to_file(out_file);
    }

    return 0;
}
