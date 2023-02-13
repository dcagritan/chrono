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
// Authors: Radu Serban, Asher Elmquist
// =============================================================================
//
// Sample test program for sequential WVP simulation.
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

#include "chrono_models/vehicle/wvp/WVP.h"

#include <chrono>
#include <thread>

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::wvp;

// =============================================================================

// Initial vehicle location and orientation
ChVector<> initLoc(0, 0, 0.5);
ChQuaternion<> initRot(1, 0, 0, 0);

// Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
VisualizationType chassis_vis_type = VisualizationType::NONE;
VisualizationType suspension_vis_type = VisualizationType::PRIMITIVES;
VisualizationType steering_vis_type = VisualizationType::PRIMITIVES;
VisualizationType wheel_vis_type = VisualizationType::NONE;
VisualizationType tire_vis_type = VisualizationType::PRIMITIVES;

// Type of tire model (RIGID, FIALA, PAC89)
TireModelType tire_model = TireModelType::PAC89;

// Type of steering model (PITMAN_ARM or PITMAN_ARM_SHAFTS)
SteeringTypeWV steering_model = SteeringTypeWV::PITMAN_ARM;

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

// Simulation step sizes
double step_size = 1e-3;
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

    // Create the vehicle, set parameters, and initialize
    WVP wvp;
    wvp.SetContactMethod(ChContactMethod::NSC);
    wvp.SetChassisFixed(false);
    wvp.SetInitPosition(ChCoordsys<>(initLoc, initRot));
    wvp.SetTireType(tire_model);
    wvp.SetTireStepSize(tire_step_size);
    wvp.setSteeringType(steering_model);
    wvp.SetInitFwdVel(0.0);
    wvp.Initialize();

    wvp.SetChassisVisualizationType(chassis_vis_type);
    wvp.SetSuspensionVisualizationType(suspension_vis_type);
    wvp.SetSteeringVisualizationType(steering_vis_type);
    wvp.SetWheelVisualizationType(wheel_vis_type);
    wvp.SetTireVisualizationType(tire_vis_type);

    {
        auto suspF = std::static_pointer_cast<WVP_DoubleWishboneFront>(wvp.GetVehicle().GetSuspension(0));
        auto springFL = suspF->GetSpring(VehicleSide::LEFT);
        auto shockFL = suspF->GetShock(VehicleSide::RIGHT);

        std::cout << "Spring rest length front: " << springFL->GetRestLength() << std::endl;
        std::cout << "Shock rest length front:  " << shockFL->GetRestLength() << std::endl;
    }
    {
        auto suspR = std::static_pointer_cast<WVP_DoubleWishboneRear>(wvp.GetVehicle().GetSuspension(1));
        auto springRL = suspR->GetSpring(VehicleSide::LEFT);
        auto shockRL = suspR->GetShock(VehicleSide::RIGHT);

        std::cout << "Spring rest length rear: " << springRL->GetRestLength() << std::endl;
        std::cout << "Shock rest length rear:  " << shockRL->GetRestLength() << std::endl;
    }

    std::cout << "Vehicle mass:               " << wvp.GetVehicle().GetMass() << std::endl;

    // ------------------
    // Create the terrain
    // ------------------

    auto patch_mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    patch_mat->SetFriction(0.9f);
    patch_mat->SetRestitution(0.01f);
    RigidTerrain terrain(wvp.GetSystem());
    auto patch = terrain.AddPatch(patch_mat, CSYSNORM, 200, 200);
    patch->SetColor(ChColor(0.8f, 0.8f, 0.5f));
    patch->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 200, 200);
    terrain.Initialize();

    // -------------------------------------
    // Create the vehicle Irrlicht interface
    // Create the driver system
    // -------------------------------------

    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("WVP sequential test");
    vis->SetChaseCamera(trackPoint, 6.0, 0.5);
    vis->Initialize();
    vis->AddTypicalLights();
    vis->AddSkyBox();
    vis->AddLogo();
    vis->AttachVehicle(&wvp.GetVehicle());

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
        double time = wvp.GetSystem()->GetChTime();

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
        wvp.Synchronize(time, driver_inputs, terrain);
        vis->Synchronize(time, driver_inputs);

        // Advance simulation for one timestep for all modules
        driver.Advance(step_size);
        terrain.Advance(step_size);
        wvp.Advance(step_size);
        vis->Advance(step_size);
        // ChQuaternion<> qW1= wvp.GetVehicle().GetWheelRot(1);
        // ChQuaternion<> qW0= wvp.GetVehicle().GetWheelRot(0);
        // ChQuaternion<> qV= wvp.GetVehicle().GetRot();
        // std::cout<<"Wheel0|"<<(qW0.Q_to_NasaAngles().z()*180/CH_C_PI) - (qV.Q_to_NasaAngles().z()*180/CH_C_PI)
        //     <<"\t|Wheel1|"<<(qW1.Q_to_NasaAngles().z()*180/CH_C_PI) - (qV.Q_to_NasaAngles().z()*180/CH_C_PI)<<std::endl;


        auto susp0 = std::static_pointer_cast<ChDoubleWishbone>(wvp.GetVehicle().GetSuspension(0));
        auto susp1 = std::static_pointer_cast<ChDoubleWishbone>(wvp.GetVehicle().GetSuspension(1));

        std::cout<<"SpringDef0|"<<susp0->GetSpringLength(LEFT)
        // <<"|force0|"<<susp0->GetSpringForce(LEFT);
        <<"|SpringDef1|"<<susp0->GetSpringLength(RIGHT)
        // <<"|force1|"<<susp0->GetSpringForce(RIGHT);
        <<"|SpringDef2|"<<susp1->GetSpringLength(LEFT)
        // <<"|force2|"<<susp1->GetSpringForce(LEFT);
        <<"|SpringDef3|"<<susp1->GetSpringLength(RIGHT)
        // <<"|force3|"<<susp1->GetSpringForce(RIGHT)
        <<std::endl;

        //ChQuaternion<> q= wvp.GetVehicle().GetRot();
        //std::cout<<"attitude|"<<q.Q_to_NasaAngles().x()<<"|bank|"<<q.Q_to_NasaAngles().y()<<"|heading|"<<q.Q_to_NasaAngles().z()<<std::endl;


        // std::cout<<wvp.GetVehicle().GetWheelPos(0).x()<<"|"<<wvp.GetVehicle().GetWheelPos(0).y()<<"|"<<wvp.GetVehicle().GetWheelPos(0).z()<<"|||"
        //     <<wvp.GetVehicle().GetWheelPos(2).x()<<"|"<<wvp.GetVehicle().GetWheelPos(2).y()<<"|"<<wvp.GetVehicle().GetWheelPos(2).z()<<std::endl;

        //log suspension locations
        /* std::cout<<"|wheel pos|"<<wvp.GetVehicle().GetSuspension(0)->GetSpindle(LEFT)->GetPos().x()-wvp.GetVehicle().GetPos().x()
          <<"|"<<wvp.GetVehicle().GetSuspension(0)->GetSpindle(LEFT)->GetPos().y()-wvp.GetVehicle().GetPos().y()
          <<"|"<<wvp.GetVehicle().GetSuspension(0)->GetSpindle(LEFT)->GetPos().z()-wvp.GetVehicle().GetPos().z()
          <<std::endl; */


        /*std::cout<<std::endl;*/
        ChCoordsys<> vehCoord = ChCoordsys<>(wvp.GetVehicle().GetPos(),wvp.GetVehicle().GetRot());
        ChVector<> vehCOM = vehCoord.TransformPointParentToLocal(wvp.GetVehicle().GetCOMFrame().GetPos());
        std::cout<<"Vehicle COM: "<<vehCOM.x()<<"|"<<vehCOM.y()<<"|"<<vehCOM.z()<<std::endl;
        /* std::cout<<"Vehicle COM|"<<wvp.GetVehicle().GetCOMFrame().GetPos().x()-wvp.GetVehicle().GetPos().x()<<"|"
            <<wvp.GetVehicle().GetCOMFrame().GetPos().y()-wvp.GetVehicle().GetPos().y()<<"|"
            <<(wvp.GetVehicle().GetCOMFrame().GetPos().z()-wvp.GetVehicle().GetPos().z());
        std::cout<<std::endl; */

        // Increment frame number
        step_number++;
    }

    return 0;
}
