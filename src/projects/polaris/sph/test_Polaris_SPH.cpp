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
// Author: Radu Serban
// =============================================================================
//
// Polaris wheeled vehicle on SPH terrain
//
// =============================================================================

#include <cstdio>
#include <string>
#include <stdexcept>
#include <iomanip>

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/ChVisualizationFsi.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"

#include "chrono_thirdparty/cxxopts/ChCLI.h"
#include "chrono_thirdparty/filesystem/path.h"

#include "DataWriter.h"
#include "CreateObjects.h"

using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::vehicle;

using std::cout;
using std::cin;
using std::endl;

// ===================================================================================================================

PolarisModel model = PolarisModel::MODIFIED;

const ChVector<> gravity(0, 0, -9.81);

// ===================================================================================================================

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir,
                     double& ramp_length,
                     double& target_speed,
                     double& tend,
                     double& step_size,
                     double& active_box_dim,
                     double& output_major_fps,
                     double& output_minor_fps,
                     int& output_frames,
                     bool& output_pos_only,
                     double& filter_window,
                     double& vis_output_fps,
                     bool& run_time_vis,
                     bool& run_time_vis_particles,
                     bool& run_time_vis_bce,
                     double& run_time_vis_fps,
                     bool& chase_cam,
                     bool& verbose);

// ===================================================================================================================

#ifdef CHRONO_OPENGL
class PolarisStats : public opengl::ChOpenGLStats {
  public:
    PolarisStats(const WheeledVehicle& vehicle)
        : ChOpenGLStats(), m_vehicle(vehicle), m_steering(0), m_throttle(0), m_braking(0) {}
    void UpdateDriverInputs(const DriverInputs& inputs) {
        m_steering = inputs.m_steering;
        m_throttle = inputs.m_throttle;
        m_braking = inputs.m_braking;
    }
    virtual void GenerateStats(ChSystem& sys) override {
        char buffer[150];
        sprintf(buffer, "TIME:     %.4f s", m_vehicle.GetChTime());
        text.Render(buffer, screen.LEFT, screen.TOP - 1 * screen.SPACING, screen.SX, screen.SY);

        sprintf(buffer, "Speed:    %.2f m/s (%.1f km/h)", m_vehicle.GetSpeed(), m_vehicle.GetSpeed() * 3.6);
        text.Render(buffer, screen.LEFT, screen.TOP - 3 * screen.SPACING, screen.SX, screen.SY);
        sprintf(buffer, "Steering: %.2f", m_steering);
        text.Render(buffer, screen.LEFT, screen.TOP - 4 * screen.SPACING, screen.SX, screen.SY);
        sprintf(buffer, "Throttle: %.2f", m_throttle);
        text.Render(buffer, screen.LEFT, screen.TOP - 5 * screen.SPACING, screen.SX, screen.SY);
        sprintf(buffer, "Braking:  %.2f", m_braking);
        text.Render(buffer, screen.LEFT, screen.TOP - 6 * screen.SPACING, screen.SX, screen.SY);
    }

    const WheeledVehicle& m_vehicle;
    double m_steering;
    double m_throttle;
    double m_braking;
};
#endif

// ===================================================================================================================

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string terrain_dir;
    double ramp_length = 0.0;
    double target_speed = 7.0;
    double tend = 30;
    double step_size = 5e-4;
    double active_box_dim = 0.5;
    double output_major_fps = 20;
    double output_minor_fps = 1000;
    int output_frames = 5;
    bool position_only = false;           // output only particle positions
    double filter_window = 0;             // do not filter data
    double vis_output_fps = 0;            // no post-processing visualization output
    bool run_time_vis = false;            // no run-time visualization
    double run_time_vis_fps = 0;          // render every simulation frame
    bool run_time_vis_particles = false;  // render only terrain surface mesh
    bool run_time_vis_bce = false;        // render vehicle meshes
    bool chase_cam = false;               // fixed camera
    bool verbose = true;

    if (!GetProblemSpecs(argc, argv, terrain_dir, ramp_length, target_speed, tend, step_size, active_box_dim,
                         output_major_fps, output_minor_fps, output_frames, position_only, filter_window,
                         vis_output_fps, run_time_vis, run_time_vis_particles, run_time_vis_bce, run_time_vis_fps,
                         chase_cam, verbose)) {
        return 1;
    }

    bool sim_output = (output_major_fps > 0);
    bool use_filter = (filter_window > 0);
    bool vis_output = (vis_output_fps > 0);

    // Check input files exist
    if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/sph_params.json")).exists()) {
        std::cout << "Input file sph_params.json not found in directory " << terrain_dir << std::endl;
        return 1;
    }
    if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/path.txt")).exists()) {
        std::cout << "Input file path.txt not found in directory " << terrain_dir << std::endl;
        return 1;
    }
    if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/particles_20mm.txt")).exists()) {
        std::cout << "Input file particles_20mm.txt not found in directory " << terrain_dir << std::endl;
        return 1;
    }
    if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/bce_20mm.txt")).exists()) {
        std::cout << "Input file bce_20mm.txt not found in directory " << terrain_dir << std::endl;
        return 1;
    }

    // Create the Chrono systems
    ChSystemNSC sys;
    ChSystemFsi sysFSI(sys);

    // Load SPH parameter file
    cout << "Load SPH parameter file..." << endl;
    sysFSI.ReadParametersFromFile(vehicle::GetDataFile(terrain_dir + "/sph_params.json"));

    sysFSI.SetActiveDomain(ChVector<>(active_box_dim, active_box_dim, 1));
    sysFSI.SetDiscreType(false, false);
    sysFSI.SetWallBC(BceVersion::ORIGINAL);
    sysFSI.SetSPHMethod(FluidDynamics::WCSPH);
    sysFSI.SetStepSize(step_size);
    sysFSI.SetVerbose(false);

    // Set simulation data output and FSI information output
    std::string out_dir = GetChronoOutputPath() + "POLARIS_SPH/";
    std::string vis_dir = out_dir + "Visualization/";
    sysFSI.SetOutputLength(0);
    ////sysFSI.SetOutputDirectory(out_dir);

    sys.Set_G_acc(gravity);

    // Create terrain
    cout << "Create terrain..." << endl;
    bool terrain_mesh_contact = false;
    auto init_pos = CreateTerrain(sys, sysFSI, terrain_dir, ramp_length, !run_time_vis_particles, terrain_mesh_contact);

    // Create vehicle
    cout << "Create vehicle..." << endl;
    auto vehicle = CreateVehicle(model, sys, init_pos);

    // Create driver
    cout << "Create path..." << endl;
    auto path = CreatePath(terrain_dir, ramp_length);
    double x_max = path->getPoint(path->getNumPoints() - 2).x() - 3.0;
    ChPathFollowerDriver driver(*vehicle, path, "my_path", target_speed);
    driver.GetSteeringController().SetLookAheadDistance(2.0);
    driver.GetSteeringController().SetGains(1.0, 0, 0);
    driver.GetSpeedController().SetGains(0.6, 0.05, 0);
    driver.Initialize();

    cout << "  Num points: " << path->getNumPoints() << endl;
    for (int i = 0; i < path->getNumPoints(); i++) {
        cout << "  [" << i << "]   " << path->getPoint(i) << endl;
    }

    #ifdef CHRONO_OPENGL
    // Create run-time visualization
    opengl::ChVisualSystemOpenGL vis;
    ChVisualizationFsi visFSI(&sysFSI, &vis);
    std::shared_ptr<ChBody> sentinel;
    std::shared_ptr<PolarisStats> stats;
    if (run_time_vis) {
        visFSI.SetTitle("Chrono::FSI single wheel demo");
        visFSI.SetSize(1280, 720);
        visFSI.SetCameraPosition(init_pos.pos + ChVector<>(-7, 0, 6), init_pos.pos + ChVector<>(1, 0, 0.5));
        visFSI.SetCameraMoveScale(1.0f);
        visFSI.EnableFluidMarkers(run_time_vis_particles);
        visFSI.EnableRigidBodyMarkers(run_time_vis_bce);
        visFSI.EnableBoundaryMarkers(false);
        visFSI.SetRenderMode(ChVisualizationFsi::RenderMode::SOLID);
        ////visFSI.SetParticleRenderMode(sysFSI.GetInitialSpacing() / 2, ChVisualizationFsi::RenderMode::SOLID);

        sentinel = CreateSentinel(sys, init_pos);

        stats = chrono_types::make_shared<PolarisStats>(*vehicle);
        vis.SetStatsRenderer(stats);
        vis.EnableStats(true);

        vis.AttachSystem(&sys);
        vis.Initialize();
    }
    #endif

    // Enable data output
    cout << "===============================================================================" << endl;
    auto sim_dir = out_dir + filesystem::path(terrain_dir).filename() + "/";
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        cout << "Error creating directory " << out_dir << endl;
        return 1;
    }
    if (!filesystem::create_directory(filesystem::path(sim_dir))) {
        cout << "Error creating directory " << sim_dir << endl;
        return 1;
    }
    DataWriterVehicle data_writer(sysFSI, vehicle);
    data_writer.SetVerbose(verbose);
    data_writer.SavePositionsOnly(position_only);
    data_writer.UseFilteredData(use_filter, filter_window);
    data_writer.Initialize(sim_dir, output_major_fps, output_minor_fps, output_frames, step_size);
    cout << "Simulation output data saved in: " << sim_dir << endl;
    cout << "===============================================================================" << endl;

    // Simulation loop
    DriverInputs driver_inputs = {0, 0, 0};
    ChTerrain terrain;

    int vis_output_steps = (int)std::round((1.0 / vis_output_fps) / step_size);
    int render_steps = (run_time_vis_fps > 0) ? (int)std::round((1.0 / run_time_vis_fps) / step_size) : 1;
    int vis_output_frame = 0;

    bool on_ramp = true;

    double t = 0;
    int frame = 0;

    while (t < tend) {
        const auto& veh_loc = vehicle->GetPos();

        // Check if vehicle approaching SPH terrain patch
        if (on_ramp && veh_loc.x() > -2) {
            // Create the wheel BCE markers at current wheel body locations
            CreateWheelBCEMarkers(vehicle, sysFSI);

            // Complete construction of FSI system
            sysFSI.Initialize();
            #ifdef CHRONO_OPENGL
            visFSI.Initialize();
            

            if (run_time_vis_bce)
                vehicle->SetTireVisualizationType(VisualizationType::NONE);
            #endif    

            on_ramp = false;
        }

        // Stop before end of patch
        if (veh_loc.x() > x_max)
            break;

        // Simulation data output
        if (sim_output && !on_ramp)
            data_writer.Process(frame);

        // Visualization data output
        if (vis_output && frame % vis_output_steps == 0) {
            if (verbose)
                cout << "Visualization output frame = " << vis_output_frame << endl;
            sysFSI.PrintParticleToFile(vis_dir);
            std::string vehicle_file = vis_dir + "/vehicle_" + std::to_string(vis_output_frame) + ".csv";
            chrono::utils::WriteVisualizationAssets(&sys, vehicle_file);

            vis_output_frame++;
        }

        // Set current driver inputs
        driver_inputs = driver.GetInputs();

        // if (t < 0.5) {
        //     driver_inputs.m_throttle = 0;
        //     driver_inputs.m_braking = 1;
        // } else {
        //     ChClampValue(driver_inputs.m_throttle, driver_inputs.m_throttle, (t - 0.5) / 0.5);
        // }

        double t_0=1.0;
        double t_1=1.5;
        double t_2=2.0;
        double t_3=3.0;
        double t_4=4.0;
        double t_5=5.0;
        double t_6=6.0;
        double t_7=7.0;

        driver_inputs.m_throttle = 0.0;
        driver_inputs.m_braking = 0.0;

        if (t < t_0)
        {
         driver_inputs.m_throttle = 0.0;
         driver_inputs.m_braking = 1.0;
        }
        else if (t < t_1)
         driver_inputs.m_throttle = (t - t_0) / (t_1-t_0);
        else if (t < t_2)
         driver_inputs.m_throttle = 1;
        else if (t < t_3)
         driver_inputs.m_throttle = (t_3 - t) / (t_3 - t_2);
        else if (t < t_4)
         driver_inputs.m_braking = 0.5*(t - t_3) / (t_4 - t_3);
        else if (t < t_6)
         driver_inputs.m_throttle = (t - t_4) / (t_6-t_4);
        else if (t < t_7)
         driver_inputs.m_braking = (t - t_6) / (tend - t_6);
        else if (t < tend)
         driver_inputs.m_throttle = 1.0;

        if (driver_inputs.m_throttle<0.00001)
         driver_inputs.m_braking = 1.0;


        ////if (verbose)
        ////    cout << std::fixed << std::setprecision(3) << "t = " << t << "  STB = " << driver_inputs.m_steering << " "
        ////         << driver_inputs.m_throttle << " " << driver_inputs.m_braking << "  spd = " << vehicle->GetSpeed()
        ////         << endl;
        
        #ifdef CHRONO_OPENGL
        // Run-time visualization
        if (run_time_vis && frame % render_steps == 0) {
            if (chase_cam) {
                ChVector<> cam_loc = veh_loc + ChVector<>(-3, 3, 2);
                ChVector<> cam_point = veh_loc;
                visFSI.SetCameraPosition(cam_loc, cam_point);
            }
            sentinel->SetPos(driver.GetSteeringController().GetTargetLocation());
            stats->UpdateDriverInputs(driver_inputs);
            if (!visFSI.Render())
                break;
        }
        #endif

        // Synchronize systems
        driver.Synchronize(t);
        vehicle->Synchronize(t, driver_inputs, terrain);

        // Advance system state
        if (on_ramp) {
            // While vehicle on acceleration ramp, only advance MBD system
            double step = std::max(step_size, 1e-3);
            driver.Advance(step);
            sys.DoStepDynamics(step);
            t += step;
        } else {
            // Advance both FSI and embedded MBD systems
            driver.Advance(step_size);
            sysFSI.DoStepDynamics_FSI();
            t += step_size;
        }

        frame++;
    }

    return 0;
}

// ===================================================================================================================

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir,
                     double& ramp_length,
                     double& target_speed,
                     double& tend,
                     double& step_size,
                     double& active_box_dim,
                     double& output_major_fps,
                     double& output_minor_fps,
                     int& output_frames,
                     bool& output_pos_only,
                     double& filter_window,
                     double& vis_output_fps,
                     bool& run_time_vis,
                     bool& run_time_vis_particles,
                     bool& run_time_vis_bce,
                     double& run_time_vis_fps,
                     bool& chase_cam,
                     bool& verbose) {
    ChCLI cli(argv[0], "Polaris SPH terrain simulation");

    cli.AddOption<std::string>("Simulation", "terrain_dir", "Directory with terrain specification data");
    cli.AddOption<double>("Simulation", "ramp_length", "Length of the acceleration ramp", std::to_string(ramp_length));
    cli.AddOption<double>("Simulation", "target_speed", "Target speed [m/s]", std::to_string(target_speed));
    cli.AddOption<double>("Simulation", "tend", "Simulation end time [s]", std::to_string(tend));
    cli.AddOption<double>("Simulation", "step_size", "Integration step size [s]", std::to_string(step_size));
    cli.AddOption<double>("Simulation", "active_box_dim", "Half-dimension of active box [m]",
                          std::to_string(active_box_dim));

    cli.AddOption<double>("Simulation output", "output_major_fps", "Simulation output major frequency [fps]",
                          std::to_string(output_major_fps));
    cli.AddOption<double>("Simulation output", "output_minor_fps", "Simulation output major frequency [fps]",
                          std::to_string(output_minor_fps));
    cli.AddOption<int>("Simulation output", "output_frames", "Number of successive output frames",
                       std::to_string(output_frames));
    cli.AddOption<bool>("Simulation output", "position_only", "Do not output particle velocities and forces");
    cli.AddOption<double>("Simulation output", "filter_window", "Running average filter window [s]",
                          std::to_string(filter_window));

    cli.AddOption<bool>("", "quiet", "Disable all messages during simulation");

    cli.AddOption<double>("Visualization", "vis_output_fps", "Visualization output frequency [fps]",
                          std::to_string(vis_output_fps));
    cli.AddOption<bool>("Visualization", "run_time_vis", "Enable run-time visualization");
    cli.AddOption<bool>("Visualization", "run_time_vis_particles", "Enable run-time particle visualization");
    cli.AddOption<bool>("Visualization", "run_time_vis_bce", "Enable run-time BCE markjer visualization");
    cli.AddOption<bool>("Visualization", "chase_cam", "Enable vehicle-following camera");
    cli.AddOption<double>("Visualization", "run_time_vis_fps", "Run-time visualization frequency [fps]",
                          std::to_string(run_time_vis_fps));

    if (!cli.Parse(argc, argv)) {
        cli.Help();
        return false;
    }

    try {
        terrain_dir = cli.Get("terrain_dir").as<std::string>();
    } catch (std::domain_error&) {
        cout << "\nERROR: Missing terrain specification directory!\n\n" << endl;
        cli.Help();
        return false;
    }

    ramp_length = cli.GetAsType<double>("ramp_length");
    target_speed = cli.GetAsType<double>("target_speed");

    output_major_fps = cli.GetAsType<double>("output_major_fps");
    output_minor_fps = cli.GetAsType<double>("output_minor_fps");
    output_frames = cli.GetAsType<int>("output_frames");
    output_pos_only = cli.GetAsType<bool>("position_only");

    filter_window = cli.GetAsType<double>("filter_window");

    vis_output_fps = cli.GetAsType<double>("vis_output_fps");
    run_time_vis = cli.GetAsType<bool>("run_time_vis");
    run_time_vis_particles = cli.GetAsType<bool>("run_time_vis_particles");
    run_time_vis_bce = cli.GetAsType<bool>("run_time_vis_bce");
    run_time_vis_fps = cli.GetAsType<double>("run_time_vis_fps");
    chase_cam = cli.GetAsType<bool>("chase_cam");

    tend = cli.GetAsType<double>("tend");
    step_size = cli.GetAsType<double>("step_size");
    active_box_dim = cli.GetAsType<double>("active_box_dim");

    verbose = !cli.GetAsType<bool>("quiet");

    return true;
}
