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
#include <fstream>
#include <array>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChParticleCloud.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/utils/ChFilters.h"
#include "chrono/assets/ChSphereShape.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono_vehicle/wheeled_vehicle/tire/RigidTire.h"

#ifdef CHRONO_OPENGL
    #include "chrono_opengl/ChVisualSystemOpenGL.h"
#endif

#include "chrono_thirdparty/cxxopts/ChCLI.h"
#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::vehicle;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

// -----------------------------------------------------------------------------

// // Speed controller target speed (in m/s)
// double target_speed = 7;


// // -----------------------------------------------------------------------------

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir,
                     double& density,
                     double& cohesion,
                     double& friction,
                     double& youngs_modulus,
                     double& poisson_ratio,
                     double& ramp_length,
                     double& target_speed,
                     double& tend,
                     double& step_size,
                     double& active_box_dim,
                     double& output_major_fps,
                     double& output_minor_fps,
                     int& output_frames,
                     int& particle_output,
                     bool& wheel_output,
                     double& filter_window_vel,
                     double& filter_window_acc,
                     double& vis_output_fps,
                     bool& run_time_vis,
                     bool& run_time_vis_particles,
                     bool& run_time_vis_bce,
                     double& run_time_vis_fps,
                     bool& chase_cam,
                     bool& verbose);

// -----------------------------------------------------------------------------

class ProxyTire : public RigidTire {
  public:
    ProxyTire(const std::string& filename) : RigidTire(filename) {}
    ProxyTire(const rapidjson::Document& d) : RigidTire(d) {}
    virtual TerrainForce ReportTireForce(ChTerrain* terrain) const override { return m_force; }
    TerrainForce m_force;
};

// -----------------------------------------------------------------------------

class NNterrain : public ChTerrain {
  public:
    NNterrain(ChSystem& sys, std::shared_ptr<WheeledVehicle> vehicle);
    // bool Load(const std::string& pt_file);
    void Create(const std::string& terrain_dir, bool vis = true);
    void Synchronize(double time, const DriverInputs& driver_inputs);
    // virtual void Advance(double step) override;

    // double GetTimerDataIn() { return m_timer_data_in(); }
    // double GetTimerModelEval() { return m_timer_model_eval(); }
    // double GetTimerDataOut() { return m_timer_data_out(); }

    void SetVerbose(bool val) { m_verbose = val; }

  private:
    ChSystem& m_sys;
    std::shared_ptr<WheeledVehicle> m_vehicle;
    std::array<std::shared_ptr<ChWheel>, 4> m_wheels;
    std::shared_ptr<ChParticleCloud> m_particles;

    ChVector<> m_box_size;
    ChVector<> m_box_offset;
    std::array<std::vector<ChAparticle*>, 4> m_wheel_particles;
    std::array<size_t, 4> m_num_particles;

    std::array<std::vector<ChVector<>>, 4> m_particle_displacements;
    std::array<TerrainForce, 4> m_tire_forces;

    ChTimer<> m_timer_data_in;
    ChTimer<> m_timer_model_eval;
    ChTimer<> m_timer_data_out;

    bool m_verbose;
};

// NNterrain::NNterrain(ChSystem& sys)
//     : m_sys(sys), m_verbose(true) {
//     // m_box_size.x() = 2.0 * std::sqrt(3.0);
//     // m_box_size.y() = 1.5;
//     // m_box_size.z() = 0.2;
//     // m_box_offset = ChVector<>(0.0, 0.0, 0.0);
// }

NNterrain::NNterrain(ChSystem& sys, std::shared_ptr<WheeledVehicle> vehicle)
    : m_sys(sys), m_vehicle(vehicle), m_verbose(true) {
    m_wheels[0] = vehicle->GetWheel(0, LEFT);
    m_wheels[1] = vehicle->GetWheel(0, RIGHT);
    m_wheels[2] = vehicle->GetWheel(1, LEFT);
    m_wheels[3] = vehicle->GetWheel(1, RIGHT);

    // Set default size and offset of sampling box
    double tire_radius = m_wheels[0]->GetTire()->GetRadius();
    double tire_width = m_wheels[0]->GetTire()->GetWidth();
    m_box_size.x() = 2.0 * std::sqrt(3.0) * tire_radius;
    m_box_size.y() = 1.5 * tire_width;
    m_box_size.z() = 0.2;
    m_box_offset = ChVector<>(0.0, 0.0, 0.0);
}



void NNterrain::Create(const std::string& terrain_dir, bool vis) {
    m_particles = chrono_types::make_shared<ChParticleCloud>();
    m_particles->SetFixed(true);
    m_particles->SetCollide(false);

    int num_particles = 0;
    ChVector<> marker;
    std::string line;
    std::string cell;

    std::ifstream is(vehicle::GetDataFile(terrain_dir + "/particles_20mm.txt"));
    getline(is, line);  // Comment line
    while (getline(is, line)) {
        std::stringstream ls(line);
        for (int i = 0; i < 3; i++) {
            getline(ls, cell, ',');
            marker[i] = stod(cell);
        }
        m_particles->AddParticle(ChCoordsys<>(marker));
        num_particles++;
        // if (num_particles > 1000000)
        if (num_particles > 10000)
            break;
    }
    is.close();

    if (vis) {
        auto sph = chrono_types::make_shared<ChSphereShape>();
        sph->GetSphereGeometry().rad = 0.01;
        m_particles->AddVisualShape(sph);
    }

    // m_sys.Add(m_particles);

    // Initial size of sampling box particle vectors
    for (int i = 0; i < 4; i++)
        m_wheel_particles[i].resize(num_particles);

}

struct in_box {
    in_box(const ChVector<>& box_pos, const ChMatrix33<>& box_rot, const ChVector<>& box_size)
        : pos(box_pos), rot(box_rot), h(box_size / 2) {}

    bool operator()(const ChAparticle* p) {
        // Convert location in box frame
        auto w = rot * (p->GetPos() - pos);

        // Check w between all box limits
        return (w.x() >= -h.x() && w.x() <= +h.x()) &&  //
               (w.y() >= -h.y() && w.y() <= +h.y()) &&  //
               (w.z() >= -h.z() && w.z() <= +h.z());
    }

    ChVector<> pos;
    ChMatrix33<> rot;
    ChVector<> h;
};

void NNterrain::Synchronize(double time, const DriverInputs& driver_inputs) {
    m_timer_data_in.reset();
    m_timer_model_eval.reset();
    m_timer_data_out.reset();

    // Prepare NN model inputs
    const auto& p_all = m_particles->GetParticles();

    m_timer_data_in.start();

    // Loop over all vehicle wheels
    std::array<ChVector<float>, 4> w_pos;
    std::array<ChQuaternion<float>, 4> w_rot;
    std::array<ChVector<float>, 4> w_nrm;
    std::array<ChVector<float>, 4> w_linvel;
    std::array<ChVector<float>, 4> w_angvel;
    std::array<bool, 4> w_contact;
    for (int i = 0; i < 4; i++) {
        // Wheel state
        const auto& w_state = m_wheels[i]->GetState();
        w_pos[i] = w_state.pos;
        w_rot[i] = w_state.rot;
        w_nrm[i] = w_state.rot.GetYaxis();
        w_linvel[i] = w_state.lin_vel;
        w_angvel[i] = w_state.ang_vel;

        auto tire_radius = m_wheels[i]->GetTire()->GetRadius();

        // Sampling OBB
        ChVector<> Z_dir(0, 0, 1);
        ChVector<> X_dir = Vcross(w_nrm[i], ChVector<>(0, 0, 1)).GetNormalized();
        ChVector<> Y_dir = Vcross(Z_dir, X_dir);
        ChMatrix33<> box_rot(X_dir, Y_dir, Z_dir);
        ChVector<> box_pos = w_pos[i] + box_rot * (m_box_offset - ChVector<>(0, 0, tire_radius));

        // Find particles in sampling OBB
        m_wheel_particles[i].resize(p_all.size());
        auto end = std::copy_if(p_all.begin(), p_all.end(), m_wheel_particles[i].begin(),
                                in_box(box_pos, box_rot, m_box_size));
        m_num_particles[i] = (size_t)(end - m_wheel_particles[i].begin());
        m_wheel_particles[i].resize(m_num_particles[i]);
        std::cout<<"i= "<<i<<"m_num_particles= "<<m_num_particles[i]<<std::endl;

        // // Do nothing if no particles under a wheel
        // if (m_num_particles[i] == 0) {
        //     std::cout<<"retrning"<<std::endl;
        //     return;
        // }

#if 0
        if (i == 0) {
            std::cout << "------------------" << std::endl;
            std::cout << part_pos << std::endl;
            std::cout << "------------------" << std::endl;
            std::cout << part_vel << std::endl;
            std::cout << "------------------" << std::endl;
            std::cout << w_pos_t << std::endl;
            std::cout << "------------------" << std::endl;
            std::cout << w_rot_t << std::endl;
            std::cout << "------------------" << std::endl;
            std::cout << w_linvel_t << std::endl;
            std::cout << "------------------" << std::endl;
            std::cout << w_angvel_t << std::endl;
            std::cout << "------------------" << std::endl;
            exit(1);
        }
#endif

#if 1
        if (m_verbose) {
            std::cout << "wheel " << i << std::endl;
            std::cout << "  num. particles: " << m_num_particles[i] << std::endl;
            std::cout << "  position:       " << w_pos[i] << std::endl;
            std::cout << "  pos. address:   " << w_pos[i].data() << std::endl;
            std::cout << "  in contact:     " << w_contact[i] << std::endl;
        }
#endif

    }


    m_timer_data_in.stop();


#if 0
    for (int i = 0; i < 4; i++) {
        std::cout << "  inputs[i]:                                            "           //
                  << &inputs[i] << std::endl;                                             //
        std::cout << "  inputs[i].toTuple()->elements()[2]:                   "           //
                  << &inputs[i].toTuple()->elements()[2] << std::endl;                    //
        std::cout << "  inputs[i].toTuple()->elements()[2].toTensor():        "           //
                  << &inputs[i].toTuple()->elements()[2].toTensor() << std::endl;         //
        std::cout << "  inputs[i].toTuple()->elements()[2].toTensor().data(): "           //
                  << &inputs[i].toTuple()->elements()[2].toTensor().data() << std::endl;  //
        const auto& wpt = inputs[i].toTuple()->elements()[2].toTensor();                  //
        std::cout << "  wheel " << i << "  position tensor: "                             //
                  << wpt[0].item<float>() << " " << wpt[1].item<float>() << " " << wpt[2].item<float>() << std::endl;
    }
#endif


    // Extract outputs

    m_timer_data_out.start();

    // Loop over all vehicle wheels
    for (int i = 0; i < 4; i++) {

        // // Extract particle displacements
        // m_particle_displacements[i].resize(m_num_particles[i]);
        // for (size_t j = 0; j < m_num_particles[i]; j++) {
        //     m_particle_displacements[i][j] =
        //         ChVector<>(0.0, 0.0, -0.01); 
        // }

        // Extract tire forces
        m_tire_forces[i].force =
            ChVector<>(0.0,0.0,0.0);
            // ChVector<>(tire_frc[0].item<float>(), tire_frc[1].item<float>(), tire_frc[2].item<float>());
        m_tire_forces[i].moment =
            ChVector<>(0.0,0.0,0.0);
            // ChVector<>(tire_frc[3].item<float>(), tire_frc[4].item<float>(), tire_frc[5].item<float>());
        m_tire_forces[i].point = ChVector<>(0, 0, 0);

        std::static_pointer_cast<ProxyTire>(m_wheels[i]->GetTire())->m_force = m_tire_forces[i];
        if (m_verbose) {
            std::cout << "  tire " << i << " force: " << m_tire_forces[i].force << std::endl;
        }
    }

    m_timer_data_out.stop();
}

// void NNterrain::Advance(double step) {
//     // Do nothing if there is at least one sampling box with no particles
//     auto product = std::accumulate(m_num_particles.begin(), m_num_particles.end(), 1, std::multiplies<int>());
//     if (product == 0)
//         return;

//     // Update state of particles in sampling boxes.
//     // Assume mass=1 for all particles.
//     double step2 = step * step / 2;
//     for (int i = 0; i < 4; i++) {
//         for (size_t j = 0; j < m_num_particles[i]; j++) {
//             auto p = m_wheel_particles[i][j]->GetPos()+m_particle_displacements[i][j];
//             m_wheel_particles[i][j]->SetPos(p);
//         }
//     }
// }

// -----------------------------------------------------------------------------

std::shared_ptr<WheeledVehicle> CreateVehicle(ChSystem& sys, const ChCoordsys<>& init_pos) {
    std::string vehicle_json = "Polaris/Polaris.json";
    ////std::string powertrain_json = "Polaris/Polaris_SimplePowertrain.json";
    std::string powertrain_json = "Polaris/Polaris_SimpleMapPowertrain.json";
    std::string tire_json = "Polaris/Polaris_RigidTire.json";

    // Create and initialize the vehicle
    auto vehicle = chrono_types::make_shared<WheeledVehicle>(&sys, vehicle::GetDataFile(vehicle_json));
    vehicle->Initialize(init_pos);
    vehicle->GetChassis()->SetFixed(false);
    vehicle->SetChassisVisualizationType(VisualizationType::MESH);
    vehicle->SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle->SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    vehicle->SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(powertrain_json));
    vehicle->InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : vehicle->GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(tire_json));
            vehicle->InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    return vehicle;
}
int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    std::string terrain_dir;

    double density = 1700;
    double cohesion = 5e3;
    double friction = 0.8;
    double youngs_modulus = 1e6;
    double poisson_ratio = 0.3;
    
    double ramp_length = 0.0;
    double target_speed = 7.0;
    double tend = 30;
    double step_size = 5e-4;
    double active_box_dim = 0.5;

    double output_major_fps = 20;
    double output_minor_fps = 1000;
    int output_frames = 5;
    int particle_output = 2;       // output all particle info
    bool wheel_output = true;      // save individual wheel output files
    double filter_window_vel = 0;  // do not filter velocity data
    double filter_window_acc = 0;  // do not filter acceleration data
    double vis_output_fps = 0;     // no post-processing visualization output

    bool run_time_vis = false;            // no run-time visualization
    double run_time_vis_fps = 0;          // render every simulation frame
    bool run_time_vis_particles = false;  // render only terrain surface mesh
    bool run_time_vis_bce = false;        // render vehicle meshes
    bool chase_cam = false;               // fixed camera

    bool verbose = true;
    bool verbose_nn = true;

    if (!GetProblemSpecs(argc, argv,                                                                        //
                         terrain_dir, density, cohesion, friction, youngs_modulus, poisson_ratio,           //
                         ramp_length, target_speed, tend, step_size, active_box_dim,                        //
                         output_major_fps, output_minor_fps, output_frames, particle_output, wheel_output,  //
                         filter_window_vel, filter_window_acc,                                              //
                         vis_output_fps,                                                                    //
                         run_time_vis, run_time_vis_particles, run_time_vis_bce, run_time_vis_fps,          //
                         chase_cam, verbose)) {
        return 1;
    }


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
    sys.Set_G_acc(ChVector<>(0, 0, -9.81));

    // Create vehicle
    cout << "Create vehicle..." << endl;
    double slope = 0;
    double banking = 0;
    if (filesystem::path(vehicle::GetDataFile(terrain_dir + "/slope.txt")).exists()) {
        std::ifstream is(vehicle::GetDataFile(terrain_dir + "/slope.txt"));
        is >> slope >> banking;
        is.close();
    }
    ChCoordsys<> init_pos(ChVector<>(4, 0, 0.02 + 4 * std::sin(slope)), Q_from_AngX(banking) * Q_from_AngY(-slope));
    auto vehicle = CreateVehicle(sys, init_pos);


    // Create terrain
    cout << "Create terrain..." << endl;
    NNterrain terrain(sys, vehicle);
    terrain.SetVerbose(verbose_nn);
    terrain.Create(terrain_dir);


    opengl::ChVisualSystemOpenGL vis;
    vis.AttachSystem(&sys);
    vis.SetWindowTitle("test_Polaris_SCMnn");
    vis.SetWindowSize(1600, 900);
    vis.SetRenderMode(opengl::WIREFRAME);
    vis.SetParticleRenderMode(0.05f, opengl::POINTS);
    vis.Initialize();
    // vis.AddCamera(ChVector<>(-2.0, 3.0, -4.0), ChVector<>(0, 0, 0));
    vis.AddCamera(ChVector<>(-3.0, 0.0, 6.0), ChVector<>(5.0, 0.0, 0.5));
    vis.SetCameraVertical(CameraVerticalDir::Z);
    // vis.SetCameraProperties(0.5f, 0.1f, 500.0f);


    // Simulation loop
    DriverInputs driver_inputs = {0, 0, 0};

    double t = 0;
    int frame = 0;

    while (t < tend) {
#ifdef CHRONO_OPENGL
        if (run_time_vis) {
            if (vis.Run()) {
                vis.BeginScene();
                vis.Render();
                vis.EndScene();
            } else {
                break;
            }
        }
#endif

        // // Stop before end of patch
        // if (vehicle->GetPos().x() > x_max)
        //     break;

        // // Set current driver inputs
        // driver_inputs = driver.GetInputs();

        // if (t < 1) {
        //     driver_inputs.m_throttle = 0;
        //     driver_inputs.m_braking = 1;
        // } else {
        //     ChClampValue(driver_inputs.m_throttle, driver_inputs.m_throttle, (t - 1) / 0.5);
        // }

        // if (verbose)
        //     cout << std::fixed << std::setprecision(3) << "t = " << t << "  STB = " << driver_inputs.m_steering << " "
        //          << driver_inputs.m_throttle << " " << driver_inputs.m_braking << "  spd = " << vehicle->GetSpeed()
        //          << "   timers = " << terrain.GetTimerDataIn() << " " << terrain.GetTimerModelEval() << " "
        //          << terrain.GetTimerDataOut() << endl;

        // Synchronize subsystems
        // driver.Synchronize(t);
        vehicle->Synchronize(t, driver_inputs, terrain);
        terrain.Synchronize(t, driver_inputs);

        // Advance system state
        // driver.Advance(step_size);
        // terrain.Advance(step_size);
        sys.DoStepDynamics(step_size);
        t += step_size;

        frame++;
    }

    return 0;
}

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir,
                     double& density,
                     double& cohesion,
                     double& friction,
                     double& youngs_modulus,
                     double& poisson_ratio,
                     double& ramp_length,
                     double& target_speed,
                     double& tend,
                     double& step_size,
                     double& active_box_dim,
                     double& output_major_fps,
                     double& output_minor_fps,
                     int& output_frames,
                     int& particle_output,
                     bool& wheel_output,
                     double& filter_window_vel,
                     double& filter_window_acc,
                     double& vis_output_fps,
                     bool& run_time_vis,
                     bool& run_time_vis_particles,
                     bool& run_time_vis_bce,
                     double& run_time_vis_fps,
                     bool& chase_cam,
                     bool& verbose) {
    ChCLI cli(argv[0], "Polaris SPH terrain simulation");

    cli.AddOption<std::string>("Problem setup", "terrain_dir", "Directory with terrain specification data");
    cli.AddOption<double>("Problem setup", "density", "Material density [kg/m3]", std::to_string(density));
    cli.AddOption<double>("Problem setup", "cohesion", "Cohesion [Pa]", std::to_string(cohesion));
    cli.AddOption<double>("Problem setup", "friction", "Coefficient of friction", std::to_string(friction));
    cli.AddOption<double>("Problem setup", "youngs_modulus", "Young's modulus [Pa]", std::to_string(youngs_modulus));
    cli.AddOption<double>("Problem setup", "poisson_ratio", "Poission ratio", std::to_string(poisson_ratio));

    cli.AddOption<double>("Simulation", "ramp_length", "Acceleration ramp length", std::to_string(ramp_length));
    cli.AddOption<double>("Simulation", "target_speed", "Target speed [m/s]", std::to_string(target_speed));
    cli.AddOption<double>("Simulation", "tend", "Simulation end time [s]", std::to_string(tend));
    cli.AddOption<double>("Simulation", "step_size", "Integration step size [s]", std::to_string(step_size));
    cli.AddOption<double>("Simulation", "active_box_dim", "Active box half-size [m]",
                          std::to_string(active_box_dim));

    cli.AddOption<double>("Simulation output", "output_major_fps", "Simulation output major frequency [fps]",
                          std::to_string(output_major_fps));
    cli.AddOption<double>("Simulation output", "output_minor_fps", "Simulation output major frequency [fps]",
                          std::to_string(output_minor_fps));
    cli.AddOption<int>("Simulation output", "output_frames", "Successive output frames",
                       std::to_string(output_frames));
    cli.AddOption<int>("Simulation output", "particle_output", "Particle output (0: none, 1: pos, 2: all)",
                       std::to_string(particle_output));
    cli.AddOption<bool>("Simulation output", "no_wheel_output", "Disable individual wheel output files");
    cli.AddOption<double>("Simulation output", "filter_window_vel", "Running average velocity filter window [s]",
                          std::to_string(filter_window_vel));
    cli.AddOption<double>("Simulation output", "filter_window_acc", "Running average acceleration filter window [s]",
                          std::to_string(filter_window_acc));

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

    density = cli.GetAsType<double>("density");
    cohesion = cli.GetAsType<double>("cohesion");
    friction = cli.GetAsType<double>("friction");
    youngs_modulus = cli.GetAsType<double>("youngs_modulus");
    poisson_ratio = cli.GetAsType<double>("poisson_ratio");

    ramp_length = cli.GetAsType<double>("ramp_length");
    target_speed = cli.GetAsType<double>("target_speed");

    output_major_fps = cli.GetAsType<double>("output_major_fps");
    output_minor_fps = cli.GetAsType<double>("output_minor_fps");
    output_frames = cli.GetAsType<int>("output_frames");
    particle_output = cli.GetAsType<int>("particle_output");
    wheel_output = !cli.GetAsType<bool>("no_wheel_output");

    filter_window_vel = cli.GetAsType<double>("filter_window_vel");
    filter_window_acc = cli.GetAsType<double>("filter_window_acc");

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

// int main(int argc, char* argv[]) {
//     GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";
// // Parse command line arguments
//     std::string terrain_dir;
//     double tend = 30.0;
//     double tenddummy = 30.0;
//     bool run_time_vis = false;
//     bool verbose = true;
//     bool verbose_nn = true;

//     if (!GetProblemSpecs(argc, argv, terrain_dir, tend, tenddummy)) {
//         return 1;
//     }

//     // Check input files exist
//     if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/sph_params.json")).exists()) {
//         std::cout << "Input file sph_params.json not found in directory " << terrain_dir << std::endl;
//         return 1;
//     }
//     // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/path.txt")).exists()) {
//     //     std::cout << "Input file path.txt not found in directory " << terrain_dir << std::endl;
//     //     return 1;
//     // }
//     // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/particles_20mm.txt")).exists()) {
//     //     std::cout << "Input file particles_20mm.txt not found in directory " << terrain_dir << std::endl;
//     //     return 1;
//     // }
//     // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/bce_20mm.txt")).exists()) {
//     //     std::cout << "Input file bce_20mm.txt not found in directory " << terrain_dir << std::endl;
//     //     return 1;
//     // }

//     // Create the Chrono systems
//     ChSystemNSC sys;


//     // std::shared_ptr<ChParticleCloud> m_particles;
//     // m_particles = chrono_types::make_shared<ChParticleCloud>();
//     // m_particles->SetFixed(true);

//     // // Create the random particles
//     // for (int np = 0; np < 100; ++np)
//     //     m_particles->AddParticle(ChCoordsys<>(ChVector<>(ChRandom() - 2, 0.5, ChRandom() + 2)));

//     // // // Mass and inertia properties.
//     // // // This will be shared among all particles in the ChParticleCloud.
//     // // particles->SetMass(0.1);
//     // // particles->SetInertiaXX(ChVector<>(0.001, 0.001, 0.001));

//     // sys.Add(m_particles);

//     // //  ==Asset== Attach a 'sphere' shape asset.. it will be used as a sample
//     // // shape to display all particles when rendering in 3D!
//     // auto sphereparticle = chrono_types::make_shared<ChSphereShape>();
//     // sphereparticle->GetSphereGeometry().rad = 0.1;
//     // m_particles->AddVisualShape(sphereparticle);


//     opengl::ChVisualSystemOpenGL vis;
//     vis.AttachSystem(&sys);
//     vis.SetWindowTitle("OpenGL assets");
//     vis.SetWindowSize(1600, 900);
//     vis.SetRenderMode(opengl::SOLID);
//     vis.SetParticleRenderMode(0.05f, opengl::POINTS);
//     vis.Initialize();
//     vis.AddCamera(ChVector<>(-2.0, 3.0, -4.0), ChVector<>(0, 0, 0));
//     vis.SetCameraVertical(CameraVerticalDir::Y);
//     vis.SetCameraProperties(0.5f, 0.1f, 500.0f);

//     double step = 0.01;
//     while (vis.Run()) {
//         vis.BeginScene();
//         vis.Render();
//         vis.EndScene();

//         sys.DoStepDynamics(step);
//     }

//     return 0;
// }

// int main(int argc, char* argv[]) {
//     // Parse command line arguments
//     std::string terrain_dir;
//     double tend = 30;
//     bool run_time_vis = false;
//     bool verbose = true;
//     bool verbose_nn = true;

//     if (!GetProblemSpecs(argc, argv, terrain_dir, tend, run_time_vis, verbose, verbose_nn)) {
//         return 1;
//     }

//     // Check input files exist
//     if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/sph_params.json")).exists()) {
//         std::cout << "Input file sph_params.json not found in directory " << terrain_dir << std::endl;
//         return 1;
//     }
//     // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/path.txt")).exists()) {
//     //     std::cout << "Input file path.txt not found in directory " << terrain_dir << std::endl;
//     //     return 1;
//     // }
//     // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/particles_20mm.txt")).exists()) {
//     //     std::cout << "Input file particles_20mm.txt not found in directory " << terrain_dir << std::endl;
//     //     return 1;
//     // }
//     // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/bce_20mm.txt")).exists()) {
//     //     std::cout << "Input file bce_20mm.txt not found in directory " << terrain_dir << std::endl;
//     //     return 1;
//     // }

//     // Create the Chrono systems
//     ChSystemNSC sys;

//     sys.Set_G_acc(ChVector<>(0, 0, -9.81));

//     // // Create vehicle
//     // cout << "Create vehicle..." << endl;
//     // double slope = 0;
//     // double banking = 0;
//     // if (filesystem::path(vehicle::GetDataFile(terrain_dir + "/slope.txt")).exists()) {
//     //     std::ifstream is(vehicle::GetDataFile(terrain_dir + "/slope.txt"));
//     //     is >> slope >> banking;
//     //     is.close();
//     // }
//     // ChCoordsys<> init_pos(ChVector<>(4, 0, 0.02 + 4 * std::sin(slope)), Q_from_AngX(banking) * Q_from_AngY(-slope));
//     // auto vehicle = CreateVehicle(sys, init_pos);

//     // // Create driver
//     // auto path = ChBezierCurve::read(vehicle::GetDataFile(terrain_dir + "/path.txt"));
//     // double x_max = path->getPoint(path->getNumPoints() - 1).x() - 3.0;
//     // ChPathFollowerDriver driver(*vehicle, path, "my_path", target_speed);
//     // driver.GetSteeringController().SetLookAheadDistance(2.0);
//     // driver.GetSteeringController().SetGains(1.0, 0, 0);
//     // driver.GetSpeedController().SetGains(0.6, 0.05, 0);
//     // driver.Initialize();

//     // Create terrain
//     cout << "Create terrain..." << endl;
//     NNterrain terrain(sys);
//     // NNterrain terrain(sys, vehicle);
//     terrain.SetVerbose(verbose_nn);
//     terrain.Create(terrain_dir,run_time_vis);

// #ifdef CHRONO_OPENGL
//     opengl::ChVisualSystemOpenGL vis;
//     if (run_time_vis) {
        
//     vis.AttachSystem(&sys);
//     vis.SetWindowTitle("OpenGL assets");
//     vis.SetWindowSize(1600, 900);
//     vis.SetRenderMode(opengl::SOLID);
//     vis.SetParticleRenderMode(0.05f, opengl::POINTS);
//     vis.Initialize();
//     vis.AddCamera(ChVector<>(-2.0, 3.0, -4.0), ChVector<>(0, 0, 0));
//     vis.SetCameraVertical(CameraVerticalDir::Y);
//     vis.SetCameraProperties(0.5f, 0.1f, 500.0f);
//     }
// #endif

//     // // Simulation loop
//     // DriverInputs driver_inputs = {0, 0, 0};

//     double step_size = 1e-3;
//     double t = 0;
//     int frame = 0;
//     while (t < tend) {
// #ifdef CHRONO_OPENGL
//         if (run_time_vis) {
//             if (vis.Run()) {
//                 vis.BeginScene();
//                 vis.Render();
//                 vis.EndScene();
//             } else {
//                 break;
//             }
//         }
// #endif

//         // Stop before end of patch
//         // if (vehicle->GetPos().x() > x_max)
//         //     break;

//         // // Set current driver inputs
//         // driver_inputs = driver.GetInputs();

//         // if (t < 1) {
//         //     driver_inputs.m_throttle = 0;
//         //     driver_inputs.m_braking = 1;
//         // } else {
//         //     ChClampValue(driver_inputs.m_throttle, driver_inputs.m_throttle, (t - 1) / 0.5);
//         // }
//         // driver_inputs.m_throttle = 0.0;
//         // driver_inputs.m_braking = 1.0;
//         // driver_inputs.m_braking = 1.0;

//         if (verbose)
//             cout << std::fixed << std::setprecision(3) << "t = " << t << std::endl;  

//         // if (verbose)
//         //     cout << std::fixed << std::setprecision(3) << "t = " << t << "  STB = " << driver_inputs.m_steering << " "
//         //          << driver_inputs.m_throttle << " " << driver_inputs.m_braking << "  spd = " << vehicle->GetSpeed()
//         //          << endl;         

//         // Synchronize subsystems
//         // driver.Synchronize(t);
//         // vehicle->Synchronize(t, driver_inputs, terrain);
//         // terrain.Synchronize(t, driver_inputs);

//         // Advance system state
//         // driver.Advance(step_size);
//         // terrain.Advance(step_size);
//         sys.DoStepDynamics(step_size);
//         t += step_size;

//         frame++;
//     }

//     return 0;
// }
