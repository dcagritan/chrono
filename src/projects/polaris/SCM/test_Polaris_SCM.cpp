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
// Author: Radu Serban
// =============================================================================
//
// Chrono::Vehicle + Chrono::Multicore demo program for simulating a HMMWV vehicle
// over rigid or granular material.
//
// Contact uses the SMC (penalty) formulation.
//
// The global reference frame has Z up.
// All units SI.
// =============================================================================

#include <cstdio>
#include <cmath>
#include <vector>

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/ChDriver.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_models/vehicle/hmmwv/HMMWV.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "DataWriter.h"
#include "CreateObjects.h"

using namespace chrono;
using namespace chrono::collision;
using namespace chrono::irrlicht;
using namespace chrono::vehicle;
using namespace chrono::vehicle::hmmwv;

using std::cout;
using std::endl;

// =============================================================================
// USER SETTINGS
// =============================================================================

// -----------------------------------------------------------------------------
// Terrain parameters
// -----------------------------------------------------------------------------

double terrainLength = 16.0;  // size in X direction
double terrainWidth = 8.0;    // size in Y direction
double delta = 0.05;          // SCM grid spacing

// -----------------------------------------------------------------------------
// Vehicle parameters
// -----------------------------------------------------------------------------

PolarisModel model = PolarisModel::MODIFIED;

// Type of tire (controls both contact and visualization)
enum class TireType { CYLINDRICAL, LUGGED };
TireType tire_type = TireType::LUGGED;

// Tire contact material properties
float Y_t = 1.0e6f;
float cr_t = 0.1f;
float mu_t = 0.8f;

// Initial vehicle position and orientation
ChVector<> initLoc(-3, 0, 0.6);
ChQuaternion<> initRot(1, 0, 0, 0);
ChCoordsys<> init_pos(initLoc, initRot);

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Simulation step size
double step_size = 2e-3;

// Time interval between two render frames (1/FPS)
double render_step_size = 2.0 / 100;

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

// Output directories
const std::string out_dir = GetChronoOutputPath() + "POLARIS_SCM";
const std::string img_dir = out_dir + "/IMG";

// Visualization output
bool img_output = false;

// =============================================================================

class MyDriver : public ChDriver {
  public:
    MyDriver(ChVehicle& vehicle, double delay) : ChDriver(vehicle), m_delay(delay) {}
    ~MyDriver() {}

    virtual void Synchronize(double time) override {
        m_throttle = 0;
        m_steering = 0;
        m_braking = 1;

        double eff_time = time - m_delay;

        // Do not generate any driver inputs for a duration equal to m_delay.
        if (eff_time < 0)
            return;

        if (eff_time > 0.2)
        { 
            m_throttle = 0.7;
            m_braking = 0;
        }
        else
        { 
            m_throttle = 3.5 * eff_time;
            m_braking = 0;
        }

        if (eff_time < 2)
            m_steering = 0;
        else
            m_steering = 0.6 * std::sin(CH_C_2PI * (eff_time - 2) / 6);
    }

  private:
    double m_delay;
};


// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";
    // Parse command line arguments
    bool verbose = true;
    bool wheel_output = true;      // save individual wheel output files
    double output_major_fps = 50;
    // --------------------
    // Create the Chrono systems
    // --------------------
    ChSystemNSC sys;
    // sys.SetNumThreads(std::min(8, ChOMP::GetNumProcs()));
    sys.SetNumThreads(1);
    const ChVector<> gravity(0, 0, -9.81);
    sys.Set_G_acc(gravity);

    // --------------------
    // Create the Polaris vehicle
    // --------------------
    cout << "Create vehicle..." << endl;
    auto vehicle = CreateVehicle(model, sys, init_pos);

    // ------------------
    // Create the terrain
    // ------------------
    // SCMDeformableTerrain terrain(system);
    SCMDeformableTerrain terrain(&sys);
    terrain.SetSoilParameters(2e6,   // Bekker Kphi
                                0,     // Bekker Kc
                                1.1,   // Bekker n exponent
                                0,     // Mohr cohesive limit (Pa)
                                30,    // Mohr friction limit (degrees)
                                0.01,  // Janosi shear coefficient (m)
                                2e8,   // Elastic stiffness (Pa/m), before plastic yield
                                3e4    // Damping (Pa s/m), proportional to negative vertical speed (optional)
    );

    ////terrain.EnableBulldozing(true);      // inflate soil at the border of the rut
    ////terrain.SetBulldozingParameters(55,   // angle of friction for erosion of displaced material at rut border
    ////                                0.8,  // displaced material vs downward pressed material.
    ////                                5,    // number of erosion refinements per timestep
    ////                                10);  // number of concentric vertex selections subject to erosion

    // Optionally, enable moving patch feature (single patch around vehicle chassis)
    // terrain.AddMovingPatch(my_hmmwv.GetChassisBody(), ChVector<>(0, 0, 0), ChVector<>(5, 3, 1));

    // Optionally, enable moving patch feature (multiple patches around each wheel)
    ////for (auto& axle : my_hmmwv.GetVehicle().GetAxles()) {
    ////    terrain.AddMovingPatch(axle->m_wheels[0]->GetSpindle(), ChVector<>(0, 0, 0), ChVector<>(1, 0.5, 1));
    ////    terrain.AddMovingPatch(axle->m_wheels[1]->GetSpindle(), ChVector<>(0, 0, 0), ChVector<>(1, 0.5, 1));
    ////}

    ////terrain.SetTexture(vehicle::GetDataFile("terrain/textures/grass.jpg"), 80, 16);
    ////terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_PRESSURE_YELD, 0, 30000.2);
    terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_SINKAGE, 0, 0.1);

    terrain.Initialize(terrainLength, terrainWidth, delta);


    // std::string vertices_filename = out_dir +  "/vertices_" + std::to_string(0) + ".csv";
    // terrain.WriteMeshVertices(vertices_filename);

    // ---------------------------------------
    // Create the vehicle Irrlicht application
    // ---------------------------------------
    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("HMMWV Deformable Soil Demo");
    vis->SetChaseCamera(trackPoint, 6.0, 0.5);
    vis->Initialize();
    vis->AddLightDirectional();
    vis->AddSkyBox();
    vis->AddLogo();
    ChVehicle* castedvehicle = vehicle.get();
    // auto castedvehicle = std::static_pointer_cast<ChVehicle>(vehicle);
    // ChVehicle* castedvehicle = (ChVehicle*)vehicle
    vis->AttachVehicle(castedvehicle);

    // --------------------
    // Create driver system
    // --------------------
    MyDriver driver(*castedvehicle, 0.5);
    driver.Initialize();

    // -----------------
    // Initialize output
    // -----------------
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cout << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    if (img_output) {
        if (!filesystem::create_directory(filesystem::path(img_dir))) {
            std::cout << "Error creating directory " << img_dir << std::endl;
            return 1;
        }
    }

    DataWriterVehicle data_writer(&sys, vehicle, terrain);
    data_writer.SetVerbose(verbose);
    data_writer.SetMBSOutput(wheel_output);
    data_writer.Initialize(out_dir, output_major_fps, step_size);
    cout << "Simulation output data saved in: " << out_dir << endl;
    cout << "===============================================================================" << endl;

    // ---------------
    // Simulation loop
    // ---------------
    std::cout << "Total vehicle mass: " << vehicle->GetMass() << std::endl;

    // Solver settings.
    sys.SetSolverMaxIterations(50);

    // Number of simulation steps between two 3D view render frames
    int render_steps = (int)std::ceil(render_step_size / step_size);

    // Initialize simulation frame counter
    int step_number = 0;
    int render_frame = 0;

    ChTimer<> timer;
    ChTimer<> timer_all;

    while (vis->Run()) {
        double time = sys.GetChTime();

        if (step_number == 800) {
            std::cout << "\nstart timers at t = " << time << std::endl;
            timer_all.start();
        }

        // Render scene
        vis->BeginScene();
        vis->Render();
        tools::drawColorbar(vis.get(), 0, 0.1, "Sinkage", 30);
        vis->EndScene();

        if (step_number % render_steps == 0) {
            // std::string vertices_filename = out_dir +  "/vertices_" + std::to_string(render_frame) + ".csv";
            // terrain.WriteMeshVertices(vertices_filename);
            // std::cout<<"Simulation time= "<<step_number*step_size<<std::endl;
            if (img_output% render_steps == 0)
            {
            char filename[100];
            sprintf(filename, "%s/img_%03d.jpg", img_dir.c_str(), render_frame + 1);
            vis->WriteImageToFile(filename);
            }
            render_frame++;
        }

        if (step_number == 800) {
            timer.start();
        }
        // // Driver inputs
        DriverInputs driver_inputs = driver.GetInputs();
        if (step_number == 800) {
            // timer_driver_GetInputs.stop();
            timer.stop();
            std::cout << "timer_driver_inputs elapsed: " << timer() << std::endl;
            timer.reset();
            timer.start();
        }
        // // Update modules
        driver.Synchronize(time);
        if (step_number == 800) {
            timer.stop();
            std::cout << "timer_driver_Synchronize elapsed: " << timer() <<std::endl;
            timer.reset();
            timer.start();
        }
        terrain.Synchronize(time);
        if (step_number == 800) {
            timer.stop();
            std::cout << "timer_terrain_Synchronize elapsed: " << timer() <<std::endl;
            timer.reset();
            timer.start();
        }
        vehicle->Synchronize(time, driver_inputs, terrain);
        if (step_number == 800) {
            timer.stop();
            std::cout << "timer_vehicle_Synchronize elapsed: " << timer() <<std::endl;
            timer.reset();
            timer.start();
        }
        vis->Synchronize("", driver_inputs);
        if (step_number == 800) {
            timer.stop();
            std::cout << "timer_vis_Synchronize elapsed: " << timer() <<std::endl;
            timer.reset();
            timer.start();
        }
        // Advance dynamics
        sys.DoStepDynamics(step_size);
        if (step_number == 800) {
            timer.stop();
            std::cout << "timer_sys_DoStepDynamics elapsed: " << timer() <<std::endl;
            timer.reset();
            timer.start();
        }
        vis->Advance(step_size);
        if (step_number == 800) {
            timer.stop();
            timer_all.stop();
            std::cout << "timer_vis_Advance elapsed: " << timer() <<std::endl;
            std::cout << "elapsed: " << timer() << std::endl;
            std::cout << "timer_all: " << timer_all() << std::endl;
            timer.reset();
            timer_all.reset();
            break;
        }
        // Increment frame number
        step_number++;
    }

    return 0;
}
