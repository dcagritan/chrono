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
// Authors: Rainer Gericke
// =============================================================================
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#include "chrono/core/ChStream.h"
#include "chrono/core/ChRealtimeStep.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/utils/ChFilters.h"

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/driver/ChIrrGuiDriver.h"
#include "chrono_vehicle/driver/ChDataDriver.h"
#include "chrono_vehicle/output/ChVehicleOutputASCII.h"

#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleIrrApp.h"

#include "chrono_models/vehicle/g-wagon/gd250.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::irrlicht;
using namespace chrono::vehicle;
using namespace chrono::vehicle::gwagon;

// =============================================================================

// Initial vehicle location and orientation
ChVector<> initLoc(0, 0, 0.4);
ChQuaternion<> initRot(1, 0, 0, 0);
// ChQuaternion<> initRot(0.866025, 0, 0, 0.5);
// ChQuaternion<> initRot(0.7071068, 0, 0, 0.7071068);
// ChQuaternion<> initRot(0.25882, 0, 0, 0.965926);
// ChQuaternion<> initRot(0, 0, 0, 1);

enum DriverMode { DEFAULT, RECORD, PLAYBACK };
DriverMode driver_mode = DEFAULT;

// Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
VisualizationType chassis_vis_type = VisualizationType::MESH;
VisualizationType suspension_vis_type = VisualizationType::PRIMITIVES;
VisualizationType steering_vis_type = VisualizationType::PRIMITIVES;
VisualizationType wheel_vis_type = VisualizationType::MESH;
VisualizationType tire_vis_type = VisualizationType::MESH;

// Collision type for chassis (PRIMITIVES, MESH, or NONE)
CollisionType chassis_collision_type = CollisionType::NONE;

// Type of powertrain model (SHAFTS, SIMPLE)
PowertrainModelType powertrain_model = PowertrainModelType::SIMPLE_MAP;

// Drive type (FWD, RWD, or AWD)
DrivelineTypeWV drive_type = DrivelineTypeWV::AWD;

// Type of tire model (TMEASY, RIGID)
TireModelType tire_model = TireModelType::TMEASY;

//
BrakeType brake_type = BrakeType::SHAFTS;

// Rigid terrain
RigidTerrain::PatchType terrain_model = RigidTerrain::PatchType::BOX;
double terrainHeight = 0;      // terrain height (FLAT terrain only)
double terrainLength = 100.0;  // size in X direction
double terrainWidth = 100.0;   // size in Y direction

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

// Contact method
ChContactMethod contact_method = ChContactMethod::SMC;
bool contact_vis = false;

// Simulation step sizes (don't increase!!)
double step_size = 3e-4;
double tire_step_size = 1e-4;

// Simulation end time
double t_end = 100;

// Time interval between two render frames
double render_step_size = 1.0 / 50;  // FPS = 50

// Output directories
const std::string out_dir = GetChronoOutputPath() + "gd250";
const std::string pov_dir = out_dir + "/POVRAY";

// Debug logging
bool debug_output = false;
double debug_step_size = 1.0 / 1;  // FPS = 1

// POV-Ray output
bool povray_output = false;

// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // --------------
    // Create systems
    // --------------

    // Create the mrole vehicle, set parameters, and initialize
    GD250 my_gd250;
    my_gd250.SetContactMethod(contact_method);
    my_gd250.SetChassisCollisionType(chassis_collision_type);
    my_gd250.SetChassisFixed(false);
    my_gd250.SetKinematicMode(false);
    my_gd250.SetLowRangeDriveline(false);
    my_gd250.SetInitPosition(ChCoordsys<>(initLoc, initRot));
    //my_gd250.SetPowertrainType(powertrain_model);
    //my_gd250.SetDriveType(drive_type);
    my_gd250.SetBrakeType(brake_type);
    my_gd250.SetTireType(tire_model);
    my_gd250.SetTireStepSize(tire_step_size);
    my_gd250.Initialize();

    if (tire_model == TireModelType::RIGID_MESH)
        tire_vis_type = VisualizationType::MESH;

    my_gd250.SetChassisVisualizationType(chassis_vis_type);
    my_gd250.SetSuspensionVisualizationType(suspension_vis_type);
    my_gd250.SetSteeringVisualizationType(steering_vis_type);
    my_gd250.SetWheelVisualizationType(wheel_vis_type);
    my_gd250.SetTireVisualizationType(tire_vis_type);

    // Create the terrain
    RigidTerrain terrain(my_gd250.GetSystem());

    MaterialInfo minfo;
    minfo.mu = 0.9f;
    minfo.cr = 0.01f;
    minfo.Y = 2e7f;
    auto patch_mat = minfo.CreateMaterial(contact_method);

    std::shared_ptr<RigidTerrain::Patch> patch;
    switch (terrain_model) {
        case RigidTerrain::PatchType::BOX:
            patch = terrain.AddPatch(patch_mat, ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), terrainLength, terrainWidth);
            patch->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 200, 200);
            break;
        case RigidTerrain::PatchType::HEIGHT_MAP:
            patch = terrain.AddPatch(patch_mat, CSYSNORM, vehicle::GetDataFile("terrain/height_maps/test64.bmp"), 128,
                                     128, 0, 4);
            patch->SetTexture(vehicle::GetDataFile("terrain/textures/grass.jpg"), 16, 16);
            break;
        case RigidTerrain::PatchType::MESH:
            patch = terrain.AddPatch(patch_mat, CSYSNORM, vehicle::GetDataFile("terrain/meshes/test.obj"));
            patch->SetTexture(vehicle::GetDataFile("terrain/textures/grass.jpg"), 100, 100);
            break;
    }
    patch->SetColor(ChColor(0.8f, 0.8f, 0.5f));

    terrain.Initialize();

    // Create the vehicle Irrlicht interface
    ChWheeledVehicleIrrApp app(&my_gd250.GetVehicle(), L"GD250 Wall2Wall Demo");
    app.SetSkyBox();
    app.AddTypicalLights(irr::core::vector3df(30.f, -30.f, 100.f), irr::core::vector3df(30.f, 50.f, 100.f), 250, 130);
    app.SetChaseCamera(trackPoint, 6.0, 0.5);
    app.SetTimestep(step_size);
    app.AssetBindAll();
    app.AssetUpdateAll();

    // -----------------
    // Initialize output
    // -----------------

    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cout << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    if (povray_output) {
        if (!filesystem::create_directory(filesystem::path(pov_dir))) {
            std::cout << "Error creating directory " << pov_dir << std::endl;
            return 1;
        }
        terrain.ExportMeshPovray(out_dir);
    }

    // Initialize output file for driver inputs
    std::string driver_file = out_dir + "/driver_inputs.txt";
    utils::CSV_writer driver_csv(" ");

    // Set up vehicle output
    my_gd250.GetVehicle().SetChassisOutput(true);
    my_gd250.GetVehicle().SetSuspensionOutput(0, true);
    my_gd250.GetVehicle().SetSteeringOutput(0, true);
    my_gd250.GetVehicle().SetOutput(ChVehicleOutput::ASCII, out_dir, "output", 0.1);

    // Generate JSON information with available output channels
    my_gd250.GetVehicle().ExportComponentList(out_dir + "/component_list.json");

    // ------------------------
    // Create the driver system
    // ------------------------

    // Create the interactive driver system
    ChIrrGuiDriver driver(app);

    // Set the time response for steering and throttle keyboard inputs.
    double steering_time = 1.0;  // time to go from 0 to +1 (or from 0 to -1)
    double throttle_time = 1.0;  // time to go from 0 to +1
    double braking_time = 0.3;   // time to go from 0 to +1
    driver.SetSteeringDelta(render_step_size / steering_time);
    driver.SetThrottleDelta(render_step_size / throttle_time);
    driver.SetBrakingDelta(render_step_size / braking_time);

    // If in playback mode, attach the data file to the driver system and
    // force it to playback the driver inputs.
    if (driver_mode == PLAYBACK) {
        driver.SetInputDataFile(driver_file);
        driver.SetInputMode(ChIrrGuiDriver::DATAFILE);
    }

    driver.Initialize();

    // ---------------
    // Simulation loop
    // ---------------

    my_gd250.GetVehicle().LogSubsystemTypes();

    if (debug_output) {
        GetLog() << "\n\n============ System Configuration ============\n";
        my_gd250.LogHardpointLocations();
    }

    // Number of simulation steps between miscellaneous events
    int render_steps = (int)std::ceil(render_step_size / step_size);
    int debug_steps = (int)std::ceil(debug_step_size / step_size);

    // Initialize simulation frame counters
    int step_number = 0;
    int render_frame = 0;

    if (contact_vis) {
        app.SetSymbolscale(1e-4);
        app.SetContactsDrawMode(IrrContactsDrawMode::CONTACT_FORCES);
    }

    ChRealtimeStepTimer realtime_timer;
    utils::ChRunningAverage RTF_filter(50);

    std::vector<double> trace_x, trace_y;

    while (app.GetDevice()->run()) {
        double time = my_gd250.GetSystem()->GetChTime();

        // End simulation
        if (time >= t_end)
            break;

        // Render scene and output POV-Ray data
        if (step_number % render_steps == 0) {
            app.BeginScene(true, true, irr::video::SColor(255, 140, 161, 192));
            app.DrawAll();
            app.EndScene();

            if (povray_output) {
                char filename[100];
                sprintf(filename, "%s/data_%03d.dat", pov_dir.c_str(), render_frame + 1);
                utils::WriteVisualizationAssets(my_gd250.GetSystem(), filename);
            }

            render_frame++;
        }

        trace_x.push_back(my_gd250.GetVehicle().GetVehiclePos().x());
        trace_y.push_back(my_gd250.GetVehicle().GetVehiclePos().y());

        // Debug logging
        if (debug_output && step_number % debug_steps == 0) {
            GetLog() << "\n\n============ System Information ============\n";
            GetLog() << "Time = " << time << "\n\n";
            my_gd250.DebugLog(OUT_SPRINGS | OUT_SHOCKS | OUT_CONSTRAINTS);

            auto marker_driver = my_gd250.GetChassis()->GetMarkers()[0]->GetAbsCoord().pos;
            auto marker_com = my_gd250.GetChassis()->GetMarkers()[1]->GetAbsCoord().pos;
            GetLog() << "Markers\n";
            std::cout << "  Driver loc:      " << marker_driver.x() << " " << marker_driver.y() << " "
                      << marker_driver.z() << std::endl;
            std::cout << "  Chassis COM loc: " << marker_com.x() << " " << marker_com.y() << " " << marker_com.z()
                      << std::endl;
        }

        // Driver inputs
        ChDriver::Inputs driver_inputs = driver.GetInputs();

        // Driver output
        if (driver_mode == RECORD) {
            driver_csv << time << driver_inputs.m_steering << driver_inputs.m_throttle << driver_inputs.m_braking
                       << std::endl;
        }

        // Update modules (process inputs from other modules)
        driver.Synchronize(time);
        terrain.Synchronize(time);
        my_gd250.Synchronize(time, driver_inputs, terrain);
        app.Synchronize(driver.GetInputModeAsString(), driver_inputs);

        // Advance simulation for one timestep for all modules
        driver.Advance(step_size);
        terrain.Advance(step_size);
        my_gd250.Advance(step_size);
        app.Advance(step_size);

        // Increment frame number
        step_number++;

        // Spin in place for real time to catch up
        realtime_timer.Spin(step_size);
        ////std::cout << RTF_filter.Add(realtime_timer.RTF) << std::endl;
    }

    if (driver_mode == RECORD) {
        driver_csv.write_to_file(driver_file);
    }

    std::ofstream trace("trace.txt");
    for (size_t i = 0; i < trace_x.size(); i++) {
        trace << trace_x[i] << "\t" << trace_y[i] << std::endl;
    }
    trace.close();
    double xmin = 9999.0;
    double xmax = -9999.0;
    double ymin = 9999.0;
    double ymax = -9999.0;
    for (size_t i = 0; i < trace_x.size(); i++) {
        if (trace_x[i] > xmax)
            xmax = trace_x[i];
        if (trace_x[i] < xmin)
            xmin = trace_x[i];
        if (trace_y[i] > ymax)
            ymax = trace_y[i];
        if (trace_y[i] < ymin)
            ymin = trace_y[i];
    }
    double delta_x = (xmax - xmin) + 3.0;
    double delta_y = (ymax - ymin) + 3.0;
    double delta = (delta_x + delta_y) / 2.0;
    std::cout << "Turn Diameter      = " << delta << " m" << std::endl;
    std::cout << "Turn Radius        = " << delta / 2.0 << " m" << std::endl;
    std::cout << "Turn Radius VehCenter = " << (delta - 3.0) / 2.0 << " m" << std::endl;
    std::cout << "Vehicle Mass       = " << my_gd250.GetVehicle().GetVehicleMass() << " kg" << std::endl;
    std::cout << "Vehicle CoG height = " << my_gd250.GetVehicle().GetVehicleCOMPos().z() << " m" << std::endl;
    return 0;
}