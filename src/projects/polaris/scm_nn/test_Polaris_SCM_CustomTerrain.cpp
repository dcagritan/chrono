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
#include "SCMTerrain_Custom.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_models/vehicle/hmmwv/HMMWV.h"

#include "chrono_thirdparty/cxxopts/ChCLI.h"
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

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir, double& tend, double& throttlemagnitude, double& steeringmagnitude, double& render_step_size, bool& heightmapterrain);

// =============================================================================
// USER SETTINGS
// =============================================================================

// -----------------------------------------------------------------------------
// Terrain parameters
// -----------------------------------------------------------------------------

// double terrainLength = 16.0;  // size in X direction
// double terrainWidth = 8.0;    // size in Y direction
// double terrainLength = 8.0;  // size in X direction
double terrainLength = 16.0;  // size in X direction
double terrainWidth = 4.0;    // size in Y direction
// double terrainLength = 32.0;  // size in X direction
// double terrainWidth = 16.0;    // size in Y direction
double delta = 0.05;          // SCM grid spacing

double throttlemagnitude=0.7;
double steeringmagnitude=0.6;
bool heightmapterrain=true;

// Initial vehicle position and orientation
// Create vehicle
ChCoordsys<> init_pos(ChVector<>(1.3, 0, 0.5), QUNIT);

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Simulation step size
double step_size = 2e-3;

// Time interval between two render frames (1/FPS)
// double render_step_size = 2.0 / 100;
double render_step_size=step_size;

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

// Output directories
const std::string out_dir = GetChronoOutputPath() + "POLARIS_SCM";
const std::string img_dir = out_dir + "/IMG";

// Visualization output
bool img_output = false;

// Vertices output
bool ver_output = true;

// =============================================================================

class MyDriver : public ChDriver {
  public:
    MyDriver(ChVehicle& vehicle, double delay) : ChDriver(vehicle), m_delay(delay) {}
    ~MyDriver() {}

    virtual void Synchronize(double time) override {
        m_throttle = 0.0;
        m_steering = 0.0;
        m_braking = 1.0;

        double eff_time = time - m_delay;

        // Do not generate any driver inputs for a duration equal to m_delay.
        if (eff_time < 0.0)
            return;

        if (eff_time > 0.0)
        { 
            //m_throttle = throttlemagnitude;
            m_throttle = throttlemagnitude * (std::sin(CH_C_2PI * (eff_time) / 2.) + 1.5);
            m_braking = 0.0;
            m_steering = steeringmagnitude * std::sin(CH_C_2PI * (eff_time) / 4.);
        }
        
        if (m_throttle > 1.)
        {
            m_throttle = 1.;
        }         
    }

  private:
    double m_delay;
};



// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";
    std::string terrain_dir;
    double tend = 5.0;
    if (!GetProblemSpecs(argc, argv,                                 
                         terrain_dir, tend, throttlemagnitude, steeringmagnitude, render_step_size, heightmapterrain)) 
    {
        return 1;
    }
    // // Check input files exist
    // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/path.txt")).exists()) {
    //     std::cout << "Input file path.txt not found in directory " << terrain_dir << std::endl;
    //     return 1;
    // }
    // Parse command line arguments
    bool verbose = true;
    bool wheel_output = true;      // save individual wheel output files
    double output_major_fps = 1.0/render_step_size;
    // --------------------
    // Create the Chrono systems
    // --------------------
    ChSystemNSC sys;
    sys.SetNumThreads(std::min(8, ChOMP::GetNumProcs()));
    const ChVector<> gravity(0, 0, -9.81);
    sys.Set_G_acc(gravity);

    // --------------------
    // Create the Polaris vehicle
    // --------------------
    cout << "Create vehicle..." << endl;
    auto vehicle = CreateVehicle(sys, init_pos);
    double x_max = (terrainLength/2.0 - 3.0);
    double y_max = (terrainWidth/2.0 - 3.0);

    // ------------------
    // Create the terrain
    // ------------------
    // SCMDeformableTerrain terrain(system);
    SCMTerrain_Custom terrain(&sys);
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
    terrain.SetPlotType(vehicle::SCMTerrain_Custom::PLOT_SINKAGE, 0, 0.1);

    if (heightmapterrain)
    { 
     terrain.Initialize(terrain_dir,  ///< [in] filename for the height map (image file)
                    terrainLength ,                       ///< [in] terrain dimension in the X direction
                    terrainWidth,                       ///< [in] terrain dimension in the Y direction
                    0.0,                        ///< [in] minimum height (black level)
                    1.5,                        ///< [in] maximum height (white level)
                    delta                        ///< [in] grid spacing (may be slightly decreased)
     );        
    }
    else
    {
     terrain.Initialize(terrainLength, terrainWidth, delta);    
    }


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
    vis->AttachVehicle(vehicle.get());

    // --------------------
    // Create driver system
    // --------------------
    MyDriver driver(*vehicle, 0.2);
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


    // Number of simulation steps between two 3D view render frames
    int render_steps = (int)std::ceil(render_step_size / step_size);

    // Initialize simulation frame counter
    int step_number = 0;
    int render_frame = 0;
    double t = 0;
    while (t < tend) {

        // const auto& veh_loc = vehicle->GetPos();
        // std::cout<<"veh_loc ="<<veh_loc<<std::endl;
        // Stop before end of patch
        // if (veh_loc.x() > x_max || veh_loc.y() > y_max )
        //     break;

        // Render scene
        vis->Run();
        vis->BeginScene();
        vis->Render();
        tools::drawColorbar(vis.get(), 0, 0.1, "Sinkage", 30);
        vis->EndScene();

        if (ver_output)
            data_writer.Process(step_number, t); 
        
        // if (step_number % render_steps == 0) {
        //     if (ver_output)
        //     {   
        //     std::string vertices_filename = out_dir +  "/vertices_" + std::to_string(render_frame) + ".csv";
        //     if (step_number==0)
        //      terrain.WriteMeshVertices(vertices_filename);
        //     else
        //      terrain.WriteMeshVerticesinz(vertices_filename);
        //     }
        //     if (img_output% render_steps == 0)
        //     {
        //     char filename[100];
        //     sprintf(filename, "%s/img_%03d.jpg", img_dir.c_str(), render_frame + 1);
        //     vis->WriteImageToFile(filename);
        //     }
        //     render_frame++;
        // }

        // // Driver inputs
        DriverInputs driver_inputs = driver.GetInputs();
        // DriverInputs driver_inputs = {0.0, 0.0, 0.0};

        // // Update modules
        driver.Synchronize(t);
        terrain.Synchronize(t);
        vehicle->Synchronize(t, driver_inputs, terrain);
        vis->Synchronize(t, driver_inputs);

        // Advance dynamics
        sys.DoStepDynamics(step_size);
        vis->Advance(step_size);
        t += step_size;

        // Increment frame number
        step_number++;

        //Pablo
        //terrain.PrintStepStatistics(cout);
    }

    return 0;
}

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir, double& tend, double& throttlemagnitude, double& steeringmagnitude, double& render_step_size, bool& heightmapterrain) 
    {
    ChCLI cli(argv[0], "Polaris SPH terrain simulation");

    cli.AddOption<std::string>("Problem setup", "terrain_dir", "Directory with terrain specification data");
    cli.AddOption<double>("Simulation", "tend", "Simulation end time [s]", std::to_string(tend));
    cli.AddOption<double>("Simulation", "throttlemagnitude", "Simulation throttle magnitude ", std::to_string(throttlemagnitude));
    cli.AddOption<double>("Simulation", "steeringmagnitude", "Simulation steering magnitude ", std::to_string(steeringmagnitude));
    cli.AddOption<double>("Simulation", "render_step_size", "Simulation render and output step size ", std::to_string(render_step_size));
    if (!cli.Parse(argc, argv)) {
        cli.Help();
        return false;
    }

    try {
        terrain_dir = cli.Get("terrain_dir").as<std::string>();
    } catch (std::domain_error&) {
        cout << "\nERROR: Missing terrain specification directory!\n\n" << endl;
        heightmapterrain=false;
        // cli.Help();
        // return false;
    }
    tend = cli.GetAsType<double>("tend");
    throttlemagnitude = cli.GetAsType<double>("throttlemagnitude");
    steeringmagnitude = cli.GetAsType<double>("steeringmagnitude");
    render_step_size = cli.GetAsType<double>("render_step_size");


    return true;
}
