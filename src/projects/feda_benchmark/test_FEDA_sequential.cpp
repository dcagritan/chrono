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
// Authors: Radu Serban, Rainer Gericke
// =============================================================================
//
// Sample test program for sequential FEDA simulation.
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
// All units SI.
//
// =============================================================================

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/driver/ChInteractiveDriverIRR.h"
#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_models/vehicle/feda/FEDA.h"
#include "chrono_models/vehicle/feda/FEDA_DoubleWishbone.h"

#include <chrono>
#include <thread>

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::feda;

// =============================================================================

// Initial vehicle location and orientation
ChVector<> initLoc(0, 0, 0.5);
ChQuaternion<> initRot(1, 0, 0, 0);

// Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
VisualizationType chassis_vis_type = VisualizationType::MESH;
VisualizationType suspension_vis_type = VisualizationType::PRIMITIVES;
VisualizationType steering_vis_type = VisualizationType::PRIMITIVES;
VisualizationType wheel_vis_type = VisualizationType::NONE;
VisualizationType tire_vis_type = VisualizationType::PRIMITIVES;

// Type of tire model (RIGID, PAC02)
TireModelType tire_model = TireModelType::PAC02;

// Type of steering model (PITMAN_ARM or PITMAN_ARM_SHAFTS)
SteeringTypeWV steering_model = SteeringTypeWV::PITMAN_ARM;

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

// Simulation step sizes
double step_size = 2e-4;
double tire_step_size = step_size;

// Simulation end time
double tend = 15;

// Time interval between two render frames
double render_step_size = 1.0 / 50;  // FPS = 50

// =============================================================================

int main(int argc, char* argv[]) {
    // --------------
    // Create systems
    // --------------

    // Parameter for aerodynamic force
    double Cd = 0.6;
    double area = 3.8;
    double air_density = 1.2041;
    // Create the vehicle, set parameters, and initialize
    FEDA feda;
    feda.SetContactMethod(ChContactMethod::NSC);
    feda.SetChassisFixed(false);
    feda.SetInitPosition(ChCoordsys<>(initLoc, initRot));
    feda.SetTireType(tire_model);
    feda.SetTireStepSize(tire_step_size);
    feda.SetAerodynamicDrag(Cd, area, air_density);
    feda.SetRideHeight_OnRoad();
    // feda.SetRideHeight_ObstacleCrossing();
    // feda.SetRideHeight_Low();

    // feda.setSteeringType(steering_model);
    feda.SetInitFwdVel(0.0);
    feda.Initialize();

    feda.SetChassisVisualizationType(chassis_vis_type);
    feda.SetSuspensionVisualizationType(suspension_vis_type);
    feda.SetSteeringVisualizationType(steering_vis_type);
    feda.SetWheelVisualizationType(wheel_vis_type);
    feda.SetTireVisualizationType(tire_vis_type);

    {
        auto suspF = std::static_pointer_cast<FEDA_DoubleWishboneFront>(feda.GetVehicle().GetSuspension(0));
        auto springFL = suspF->GetSpring(VehicleSide::LEFT);
        auto shockFL = suspF->GetShock(VehicleSide::RIGHT);

        // std::cout << "Spring rest length front: " << springFL->GetSpringRestLength() << std::endl;
        // std::cout << "Shock rest length front:  " << shockFL->GetSpringRestLength() << std::endl;
    }
    {
        auto suspR = std::static_pointer_cast<FEDA_DoubleWishboneRear>(feda.GetVehicle().GetSuspension(1));
        auto springRL = suspR->GetSpring(VehicleSide::LEFT);
        auto shockRL = suspR->GetShock(VehicleSide::RIGHT);

        // std::cout << "Spring rest length rear: " << springRL->GetSpringRestLength() << std::endl;
        // std::cout << "Shock rest length rear:  " << shockRL->GetSpringRestLength() << std::endl;
    }

    std::cout << "Vehicle mass:               " << feda.GetVehicle().GetMass() << std::endl;

    // ------------------
    // Create the terrain
    // ------------------

    auto patch_mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    patch_mat->SetFriction(0.8f);
    patch_mat->SetRestitution(0.01f);
    RigidTerrain terrain(feda.GetSystem());
    auto patch = terrain.AddPatch(patch_mat, CSYSNORM, 200, 200);
    patch->SetColor(ChColor(0.8f, 0.8f, 0.5f));
    patch->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 200, 200);
    terrain.Initialize();

    // -------------------------------------
    // Create the vehicle Irrlicht interface
    // Create the driver system
    // -------------------------------------

    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("FED Alpha sequential test");
    vis->SetChaseCamera(trackPoint, 8.0, 0.5);
    vis->SetHUDLocation(500, 20);
    vis->Initialize();
    vis->AddTypicalLights();
    vis->AddSkyBox();
    vis->AddLogo();
    vis->AttachVehicle(&feda.GetVehicle());

    // Create the interactive driver system
    ChInteractiveDriverIRR driver(*vis);

    // Set the time response for steering and throttle keyboard inputs.
    double steering_time = 1.0;  // time to go from 0 to +1 (or from 0 to -1)
    double throttle_time = 1.0;  // time to go from 0 to +1
    double braking_time = 0.3;   // time to go from 0 to +1
    driver.SetSteeringDelta(render_step_size / steering_time);
    driver.SetThrottleDelta(render_step_size / throttle_time);
    driver.SetBrakingDelta(render_step_size / braking_time);

    driver.Initialize();

    // ---------------
    // Simulation loop
    // ---------------

    int render_steps = (int)std::ceil(render_step_size / step_size);
    int step_number = 0;

    while (vis->Run()) {
        double time = feda.GetSystem()->GetChTime();

        // Render scene
        if (step_number % render_steps == 0) {
            vis->BeginScene();
            vis->Render();
            vis->EndScene();
        }

        // Driver inputs
        DriverInputs driver_inputs = driver.GetInputs();

        // Update modules (process inputs from other modules)
        driver.Synchronize(time);
        terrain.Synchronize(time);
        feda.Synchronize(time, driver_inputs, terrain);
        vis->Synchronize(time, driver_inputs);

        // Advance simulation for one timestep for all modules
        driver.Advance(step_size);
        terrain.Advance(step_size);
        feda.Advance(step_size);
        vis->Advance(step_size);
        // ChQuaternion<> qW1= feda.GetVehicle().GetWheelRot(1);
        // ChQuaternion<> qW0= feda.GetVehicle().GetWheelRot(0);
        // ChQuaternion<> qV= feda.GetVehicle().GetRot();
        // std::cout<<"Wheel0|"<<(qW0.Q_to_NasaAngles().z()*180/CH_C_PI) - (qV.Q_to_NasaAngles().z()*180/CH_C_PI)
        //     <<"\t|Wheel1|"<<(qW1.Q_to_NasaAngles().z()*180/CH_C_PI) -
        //     (qV.Q_to_NasaAngles().z()*180/CH_C_PI)<<std::endl;

        auto susp0 = std::static_pointer_cast<ChDoubleWishbone>(feda.GetVehicle().GetSuspension(0));
        auto susp1 = std::static_pointer_cast<ChDoubleWishbone>(feda.GetVehicle().GetSuspension(1));

        ////auto susp = feda.GetVehicle().GetAxle(0)->GetSuspension();
        ////auto states = (std::static_pointer_cast<FEDA_DoubleWishboneFront>(susp))->GetShock(LEFT)->GetStates();
        ////std::cout << time << "   " << states(0) << "  " << states(1) << std::endl;

        ChCoordsys<> vehCoord = ChCoordsys<>(feda.GetVehicle().GetPos(), feda.GetVehicle().GetRot());
        ChVector<> vehCOM = vehCoord.TransformPointParentToLocal(feda.GetVehicle().GetCOMFrame().GetPos());
        // std::cout << "Vehicle COM: " << vehCOM.x() << "|" << vehCOM.y() << "|" << vehCOM.z() << std::endl;
        /* std::cout<<"Vehicle
        COM|"<<feda.GetVehicle().GetCOMFrame().GetPos().x()-feda.GetVehicle().GetPos().x()<<"|"
            <<feda.GetVehicle().GetCOMFrame().GetPos().y()-feda.GetVehicle().GetPos().y()<<"|"
            <<(feda.GetVehicle().GetCOMFrame().GetPos().z()-feda.GetVehicle().GetPos().z());
        std::cout<<std::endl; */

        // Increment frame number
        step_number++;
    }

    GetLog() << "CoG of the FED alpha above ground = " << feda.GetVehicle().GetCOMFrame().GetPos().z() << " m\n";
    // Ride height test, originally the tests where made with body reference points, we don't have it
    // Ride height OnRoad is reached when the drive shafts are straight lines
    double refHeightFront =
        (feda.GetVehicle().GetWheel(0, LEFT)->GetPos().z() + feda.GetVehicle().GetWheel(0, RIGHT)->GetPos().z()) / 2.0;
    double refHeightRear =
        (feda.GetVehicle().GetWheel(1, LEFT)->GetPos().z() + feda.GetVehicle().GetWheel(1, RIGHT)->GetPos().z()) / 2.0;
    double bodyHeightFront = feda.GetVehicle().GetPointLocation(ChVector<>(0, 0, 0)).z();
    double bodyHeightRear = feda.GetVehicle().GetPointLocation(ChVector<>(-3.302, 0, 0)).z();
    double rideHeightFront = bodyHeightFront - refHeightFront;
    double rideHeightRear = bodyHeightRear - refHeightRear;
    GetLog() << "Ride Height Front = " << rideHeightFront << " m, Rear = " << rideHeightRear << " m\n";
    return 0;
}
