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

//#define USE_IRRLICHT


#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"

#ifdef USE_IRRLICHT
#include "chrono_vehicle/driver/ChIrrGuiDriver.h"
#endif

#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleIrrApp.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_models/vehicle/wvp/WVP_FollowerDataDriver.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/core/ChFileutils.h"

#include "chrono_models/vehicle/wvp/WVP.h"

#include <chrono>
#include <thread>

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::wvp;

// =============================================================================

// Initial vehicle location and orientation
ChVector<> initLoc(-60.96, 0, 0.5);
ChQuaternion<> initRot = Q_from_AngZ(-CH_C_PI/2.0);//(1, 0, 0, 0);

// Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
VisualizationType chassis_vis_type = VisualizationType::NONE;
VisualizationType suspension_vis_type = VisualizationType::PRIMITIVES;
VisualizationType steering_vis_type = VisualizationType::PRIMITIVES;
VisualizationType wheel_vis_type = VisualizationType::NONE;
VisualizationType tire_vis_type = VisualizationType::PRIMITIVES;

// Type of tire model (RIGID, FIALA, PAC89)
TireModelType tire_model = TireModelType::PAC89;

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

// Simulation step sizes
double step_size = 1e-4;
double tire_step_size = step_size;

// Simulation end time
double tend = 15;

// Time interval between two render frames
double render_step_size = 1.0 / 50;  // FPS = 50
double output_step_size = 1e-2;

//output directory
const std::string out_dir = "../WVP_CONST_TURN";
const std::string pov_dir = out_dir + "/POVRAY";
bool povray_output = false;
bool data_output = true;

//vehicle driver inputs
// Desired vehicle speed (m/s)
double mph_to_ms = 0.44704;
double target_speed = 0*mph_to_ms;

// std::string path_file("paths/NATO_double_lane_change.txt");
std::string path_file("paths/circlepath_200ft_radius.txt");
std::string path_fileCW("paths/circlepath_200ft_radius.txt");
std::string path_fileCCW("paths/circlepath_200ft_radiusCCW.txt");


std::string steering_controller_file("wvp/SteeringController.json");
std::string speed_controller_file("wvp/SpeedController.json");


std::string output_file_name("DLC_CONST_TURN_");


// =============================================================================

int main(int argc, char* argv[]) {
    //send in args: filename time speed LtR(or RtL)
    //read from args
    if(argc > 2){
        //std::cout<<argv[0]<<"|"<<argv[1]<<"|"<<argv[2]<<"|"<<argv[3]<<std::endl;
        tend = atof(argv[1]);
        if(atof(argv[2])==0){
            path_file = path_fileCW;
            output_file_name += "CW";
            initRot= Q_from_AngZ(-CH_C_PI/2.0);
        }   
        else if(atof(argv[2])==1){ //0 = CW, 1=CCW
            path_file = path_fileCCW;
            output_file_name += "CCW";
            initRot= Q_from_AngZ(CH_C_PI/2.0);
        }
    }


    // --------------
    // Create systems
    // --------------

    // Create the vehicle, set parameters, and initialize
    WVP wvp;
    wvp.SetChassisFixed(false);
    wvp.SetInitPosition(ChCoordsys<>(initLoc, initRot));
    wvp.SetTireType(tire_model);
    wvp.SetTireStepSize(tire_step_size);
    wvp.Initialize();

    wvp.SetChassisVisualizationType(chassis_vis_type);
    wvp.SetSuspensionVisualizationType(suspension_vis_type);
    wvp.SetSteeringVisualizationType(steering_vis_type);
    wvp.SetWheelVisualizationType(wheel_vis_type);
    wvp.SetTireVisualizationType(tire_vis_type);

    std::cout << "Total vehicle mass: " << wvp.GetVehicle().GetVehicleMass() << std::endl;

    // ------------------
    // Create the terrain
    // ------------------

    RigidTerrain terrain(wvp.GetSystem());
    terrain.SetContactFrictionCoefficient(0.8f);
    terrain.SetContactRestitutionCoefficient(0.01f);
    terrain.SetContactMaterialProperties(2e7f, 0.3f);
    terrain.SetColor(ChColor(0.8f, 0.8f, 0.5f));
    terrain.SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 20, 20);
    terrain.Initialize(0, 500, 500);

    //create the driver -> path follower
    auto path = ChBezierCurve::read(vehicle::GetDataFile(path_file));
    ChPathFollowerDriver driver(wvp.GetVehicle(), vehicle::GetDataFile(steering_controller_file),
                                vehicle::GetDataFile(speed_controller_file), path, "my_path", target_speed, true);

    driver.Initialize();


#ifdef USE_IRRLICHT
    // -------------------------------------
    // Create the vehicle Irrlicht interface
    // Create the driver system
    // -------------------------------------

    ChWheeledVehicleIrrApp app(&wvp.GetVehicle(), &wvp.GetPowertrain(), L"WVP sequential test");
    app.SetSkyBox();
    app.AddTypicalLights(irr::core::vector3df(30.f, -30.f, 100.f), irr::core::vector3df(30.f, 50.f, 100.f), 250, 130);
    app.SetChaseCamera(trackPoint, 6.0, 0.5);
    /*app.SetTimestep(step_size);*/
    app.AssetBindAll();
    app.AssetUpdateAll();

    // Visualization of controller points (sentinel & target)
    irr::scene::IMeshSceneNode* ballS = app.GetSceneManager()->addSphereSceneNode(0.1f);
    irr::scene::IMeshSceneNode* ballT = app.GetSceneManager()->addSphereSceneNode(0.1f);
    ballS->getMaterial(0).EmissiveColor = irr::video::SColor(0, 255, 0, 0);
    ballT->getMaterial(0).EmissiveColor = irr::video::SColor(0, 0, 255, 0);
#endif
    
    // -------------
    // Prepare output
    // -------------

    if (data_output || povray_output) {
        if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
            std::cout << "Error creating directory " << out_dir << std::endl;
            return 1;
        }
    }
    if (povray_output) {
        if (ChFileutils::MakeDirectory(pov_dir.c_str()) < 0) {
            std::cout << "Error creating directory " << pov_dir << std::endl;
            return 1;
        }
        driver.ExportPathPovray(out_dir);
    }

    utils::CSV_writer csv("\t");
    csv.stream().setf(std::ios::scientific | std::ios::showpos);
    csv.stream().precision(6);

    // Number of simulation steps between two 3D view render frames
    int render_steps = (int)std::ceil(render_step_size / step_size);

    // Number of simulation steps between two output frames
    int output_steps = (int)std::ceil(output_step_size / step_size);

    // ---------------
    // Simulation loop
    // ---------------

    std::cout<<"data at: "<<vehicle::GetDataFile(steering_controller_file)<<std::endl;
    std::cout<<"data at: "<<vehicle::GetDataFile(speed_controller_file)<<std::endl;
    std::cout<<"data at: "<<vehicle::GetDataFile(path_file)<<std::endl;

    int step_number = 0;
    int render_frame = 0;

    double time = 0;

    //output headings for the saved data file
    csv <<"time"<<"throttle"<<"target_speed"<<"SteeringAngle"<<"vehicleSpeed";

    csv <<"ChassisAttitude"<<"ChassisBank"<<"ChassisHeading";
    csv <<"LateralAcceleration";

    csv <<"W0Pos x"<<"W0PosY"<<"W0PosZ"<<"W1Posx"<<"W1PosY"<<"W1PosZ";
    csv <<"W2Pos x"<<"W2PosY"<<"W2PosZ"<<"W3Posx"<<"W3PosY"<<"W3PosZ"<< std::endl;


#ifdef USE_IRRLICHT
    while (app.GetDevice()->run()) {
        

        //path visualization
        const ChVector<>& pS = driver.GetSteeringController().GetSentinelLocation();
        const ChVector<>& pT = driver.GetSteeringController().GetTargetLocation();
        ballS->setPosition(irr::core::vector3df((irr::f32)pS.x(), (irr::f32)pS.y(), (irr::f32)pS.z()));
        ballT->setPosition(irr::core::vector3df((irr::f32)pT.x(), (irr::f32)pT.y(), (irr::f32)pT.z()));

        /*std::cout<<"Target:\t"<<(irr::f32)pT.x()<<",\t "<<(irr::f32)pT.y()<<",\t "<<(irr::f32)pT.z()<<std::endl;
        std::cout<<"Vehicle:\t"<<wvp.GetVehicle().GetChassisBody()->GetPos().x()
          <<",\t "<<wvp.GetVehicle().GetChassisBody()->GetPos().y()<<",\t "
          <<wvp.GetVehicle().GetChassisBody()->GetPos().z()<<std::endl;*/

        // Render scene
        if (step_number % render_steps == 0) {
            app.BeginScene(true, true, irr::video::SColor(255, 140, 161, 192));
            app.DrawAll();
            app.EndScene();
        }
#else
    while(wvp.GetSystem()->GetChTime() < tend){
#endif

        time = wvp.GetSystem()->GetChTime();

        //bump up speed by 5 mph every 20 seconds
        int timMult = (int) (time/step_size);
        int interval = (int) (20 / step_size);
        if(timMult%interval == 0){
            target_speed+=5.0*mph_to_ms;
            driver.SetDesiredSpeed(target_speed);
        }



        // Collect output data from modules (for inter-module communication)
        double throttle_input = driver.GetThrottle();
        double steering_input = driver.GetSteering();
        double braking_input = driver.GetBraking();

        // Update modules (process inputs from other modules)
        driver.Synchronize(time);
        terrain.Synchronize(time);
        wvp.Synchronize(time, steering_input, braking_input, throttle_input, terrain);
#ifdef USE_IRRLICHT
        app.Synchronize("Follower driver", steering_input, throttle_input, braking_input);
#endif

        // Advance simulation for one timestep for all modules

        driver.Advance(step_size);
        terrain.Advance(step_size);
        wvp.Advance(step_size);
#ifdef USE_IRRLICHT
        app.Advance(step_size);
#endif

        if (povray_output && step_number % render_steps == 0) {
            char filename[100];
            sprintf(filename, "%s/data_%03d.dat", pov_dir.c_str(), render_frame + 1);
            utils::WriteShapesPovray(wvp.GetSystem(), filename);
            render_frame++;

        }

        if(data_output && step_number % output_steps == 0){
            //output time to check simulation is running
            std::cout<<time<<"|"<<target_speed<<std::endl;


            csv <<time<<throttle_input<<target_speed<<steering_input<<wvp.GetVehicle().GetVehicleSpeed();
            ChQuaternion<> q= wvp.GetVehicle().GetVehicleRot();

            csv << q.Q_to_NasaAngles();

            csv << wvp.GetVehicle().GetVehicleAcceleration(wvp.GetVehicle().GetChassis()->GetCOMPos()).y();

            for(int i=0;i<4;i++){
                csv << wvp.GetVehicle().GetWheelPos(i);
            }
            csv << std::endl;

        }


        // std::cout<<time<<std::endl;
        // Increment frame number
        step_number++;
    }

    if (data_output) {
        csv.write_to_file(out_dir + "/" + output_file_name + ".dat");
    }

    return 0;
}