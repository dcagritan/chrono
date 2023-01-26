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
// Authors: Radu Serban, Rainer Gericke
// =============================================================================
//
// Main driver function for a mrole specified through JSON files + example for
// obtaining shock effect results presented by ISO 2631-5
//
// Halfround shaped obstacles
//
// If using the Irrlicht interface, driver inputs are obtained from the keyboard.
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#include "chrono/core/ChStream.h"
#include "chrono/utils/ChFilters.h"

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/powertrain/SimplePowertrain.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/terrain/CRGTerrain.h"

#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"

#include "chrono_models/vehicle/g-wagon/gd250.h"

#ifdef CHRONO_IRRLICHT
    #include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"
    // specify whether the demo should actually use Irrlicht
    #define USE_IRRLICHT
#endif

// =============================================================================

using namespace chrono;
using namespace chrono::utils;
using namespace chrono::vehicle;
using namespace chrono::vehicle::gwagon;

// Tire collision type
ChTire::CollisionType collision_type = ChTire::CollisionType::ENVELOPE;

// Type of tire model (RIGID, TMEASY)
TireModelType tire_model = TireModelType::TMEASY;

// Driver input files
std::string path_file("paths/straightOrigin.txt");
std::string steering_controller_file("hmmwv/SteeringController.json");
std::string speed_controller_file("hmmwv/SpeedController.json");

// Initial vehicle position
ChVector<> initLoc(20, 0, 0.4);

// Simulation step size (should not be too high!)
double step_size = 1e-3;

const double mph_to_ms = 0.44704;

// Road visualization (mesh or boundary lines)
bool useMesh = true;

// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2018 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    const int heightVals[16] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    double az_limit = 2.5;
    double az;

    int iObstacle = 1;
    double target_speed = 5.0;
    int step_speed = 5.0;
    int start_speed = 10.0;
    int end_speed = 10.0;
    // CRG files for terrain
    std::string crg_terrain_file("terrain/crg_roads/halfround_2in.crg");

    ChFunction_Recorder resLogger;
    resLogger.AddPoint(0, 0);  // needed, if the first simulation run is beyond limit

    if (argc != 5) {
        GetLog() << "usage: demo_VEH_Shock ObstacleNumber StartSpeed_mph EndSpeed_mph StepSpeed_mph\n\n";
        return 1;
    }
    iObstacle = ChClamp(atoi(argv[1]), 1, 15);
    start_speed = atoi(argv[2]);
    end_speed = atoi(argv[3]);
    step_speed = atoi(argv[4]);
    for (int iSpeed = start_speed; iSpeed <= end_speed; iSpeed += step_speed) {
        target_speed = iSpeed;
        switch (iObstacle) {
            case 2:
                crg_terrain_file = "terrain/crg_roads/halfround_4in.crg";
                break;
            case 3:
                crg_terrain_file = "terrain/crg_roads/halfround_6in.crg";
                break;
            case 4:
                crg_terrain_file = "terrain/crg_roads/halfround_8in.crg";
                break;
            case 5:
                crg_terrain_file = "terrain/crg_roads/halfround_10in.crg";
                break;
            case 6:
                crg_terrain_file = "terrain/crg_roads/halfround_12in.crg";
                break;
            case 7:
                crg_terrain_file = "terrain/crg_roads/halfround_14in.crg";
                break;
            case 8:
                crg_terrain_file = "terrain/crg_roads/halfround_16in.crg";
                break;
            case 9:
                crg_terrain_file = "terrain/crg_roads/halfround_18in.crg";
                break;
            case 10:
                crg_terrain_file = "terrain/crg_roads/halfround_20in.crg";
                break;
            case 11:
                crg_terrain_file = "terrain/crg_roads/halfround_22in.crg";
                break;
            case 12:
                crg_terrain_file = "terrain/crg_roads/halfround_24in.crg";
                break;
            case 13:
                crg_terrain_file = "terrain/crg_roads/halfround_26in.crg";
                break;
            case 14:
                crg_terrain_file = "terrain/crg_roads/halfround_28in.crg";
                break;
            case 15:
                crg_terrain_file = "terrain/crg_roads/halfround_30in.crg";
                break;
        }
        // --------------------------
        // Create the various modules
        // --------------------------

        if (target_speed > 20) initLoc.x() -= 20;
        if (target_speed > 30) initLoc.x() -= 20;
        if (target_speed > 40) initLoc.x() -= 40;
        if (target_speed > 50) initLoc.x() -= 60;
        if (target_speed > 60) initLoc.x() -= 100;

        // Create the vehicle system
        GD250 gd250;
        gd250.SetContactMethod(ChContactMethod::NSC);
        gd250.SetChassisFixed(false);
        gd250.SetKinematicMode(true);
        gd250.SetInitPosition(ChCoordsys<>(initLoc, QUNIT));
        gd250.SetTireType(tire_model);
        //gd250.SetTireCollisionType(ChTire::CollisionType::ENVELOPE);
        gd250.SetInitFwdVel(target_speed * mph_to_ms);
        gd250.Initialize();

        gd250.SetChassisVisualizationType(VisualizationType::NONE);
        gd250.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
        gd250.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
        gd250.SetWheelVisualizationType(VisualizationType::NONE);
        gd250.SetTireVisualizationType(VisualizationType::MESH);
        ////gd250.GetChassis()->SetFixed(true);

        GetLog() << "\nBe patient - startup may take some time... \n";

        // Create the ground
        // RigidTerrain terrain(gd250.GetSystem(), vehicle::GetDataFile(rigidterrain_file));

        CRGTerrain terrain(gd250.GetSystem());
        terrain.UseMeshVisualization(useMesh);
        terrain.SetContactFrictionCoefficient(0.8f);
        terrain.Initialize(vehicle::GetDataFile(crg_terrain_file));

        ChISO2631_Shock_SeatCushionLogger seat_logger(step_size);

        // Create the driver
        auto path = ChBezierCurve::read(vehicle::GetDataFile(path_file));
        ChPathFollowerDriver driver(gd250.GetVehicle(), vehicle::GetDataFile(steering_controller_file),
                                    vehicle::GetDataFile(speed_controller_file), path, "my_path",
                                    target_speed * mph_to_ms);
        driver.Initialize();

#ifdef USE_IRRLICHT
        auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
        vis->SetWindowTitle("Mercedes GD250 Shock Test");
        vis->SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), 6.0, 0.5);
        vis->Initialize();
        vis->AddTypicalLights();
        vis->AddSkyBox();
        vis->AddLogo();
        vis->AttachVehicle(&gd250.GetVehicle());
#endif

        // ---------------
        // Simulation loop
        // ---------------

        // Logging of seat acceleration data on flat road surface is useless
        double xstart = 40.0;  // start logging when the gd250 crosses this x position
        double xend = 100.0;   // end logging here, this also the end of our world

#ifdef USE_IRRLICHT

        while (vis->Run()) {
            // Render scene
            vis->BeginScene();
            vis->Render();

            // Driver inputs
            DriverInputs driver_inputs = driver.GetInputs();

            // Update modules (process inputs from other modules)
            double time = gd250.GetSystem()->GetChTime();
            driver.Synchronize(time);
            gd250.Synchronize(time, driver_inputs, terrain);
            terrain.Synchronize(time);
            vis->Synchronize("", driver_inputs);

            // Advance simulation for one timestep for all modules
            driver.Advance(step_size);
            gd250.Advance(step_size);
            terrain.Advance(step_size);
            vis->Advance(step_size);

            double xpos = gd250.GetVehicle().GetPos().x();
            if (xpos >= xend) {
                break;
            }
            if (xpos >= xstart) {
                ChVector<> seat_acc = gd250.GetVehicle().GetPointAcceleration(
                        gd250.GetVehicle().GetChassis()->GetLocalDriverCoordsys().pos);
                seat_logger.AddData(seat_acc);
            }

            vis->EndScene();
        }

#else
        const double ms_to_mph = 2.2369362921;

        double v_pos;
        while ((v_pos = gd250.GetVehicle().GetPos().x()) < xend) {
            // Driver inputs
            DriverInputs driver_inputs = driver.GetInputs();

            // Update modules (process inputs from other modules)
            double time = gd250.GetSystem()->GetChTime();
            driver.Synchronize(time);
            gd250.Synchronize(time, driver_inputs, terrain);
            terrain.Synchronize(time);

            // Advance simulation for one timestep for all modules
            driver.Advance(step_size);
            gd250.Advance(step_size);
            terrain.Advance(step_size);

            if (v_pos >= xstart) {
                double speed = gd250.GetSpeed();
                ChVector<> seat_acc = gd250.GetPointAcceleration(gd250.GetChassis()->GetLocalDriverCoordsys().pos);
                seat_logger.AddData(seat_acc);
            }
        }

#endif

        const double ms_to_mph = 2.2369362921;
        double se_low = 0.5;
        double se_high = 0.8;
        double se = seat_logger.GetSe();

        double az_limit = 2.5;
        double az = seat_logger.GetLegacyAz();
        resLogger.AddPoint(az, target_speed);
        if(az >= 2.5) {
            break;
        }
        GetLog() << "Shock Simulation Results #1 (ISO 2631-5 Method):\n";
        GetLog() << "  Significant Speed                        Vsig = " << target_speed << " m/s\n";
        GetLog() << "  Equivalent Static Spine Compressive Stress Se = " << se << " MPa\n";
        if (se <= se_low) {
            GetLog() << "Se <= " << se_low
                     << " MPa (ok) - low risc of health effect, below limit for average occupants\n";
        } else if (se >= se_high) {
            GetLog() << "Se >= " << se_high << " MPa - severe risc of health effect!\n";
        } else {
            GetLog() << "Se is between [" << se_low << ";" << se_high << "] - risc of health effects, above limit!\n";
        }
        GetLog() << "\nShock Simulation Results #2 (Traditional NRMM Method):\n";
        GetLog() << "Significant Speed = " << target_speed << " mph\n";
        GetLog() << "Obstacle Height   = " << heightVals[iObstacle] << " in\n";
        GetLog() << "  Maximum Vertical Seat Acceleration = " << az << " g\n";
        if (az <= az_limit) {
            GetLog() << "Az <= " << az_limit << " g (ok)\n";
        } else {
            GetLog() << "Az > " << az_limit << " g - severe risk for average occupant!\n";
        }
    }
    double amin, amax;
    double resSpeed = end_speed;
    resLogger.Estimate_x_range(amin, amax);
    if (amax >= 2.5) {
        resSpeed = resLogger.Get_y(2.5);
    }
    GetLog() << "ObsHeight       = " << heightVals[iObstacle] << " in\n";
    GetLog() << "Speed at 2.5g   = " << resSpeed << " mph\n";
    GetLog() << "Max. vert. Acc. = " << amax << " g\n";

    return 0;
}