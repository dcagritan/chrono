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

#include <torch/torch.h>
#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <torchcluster/cluster.h>

using namespace chrono;
using namespace chrono::vehicle;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

// -----------------------------------------------------------------------------

// Speed controller target speed (in m/s)
double target_speed = 7;

// NN model
std::string NN_module_name = "terrain/sph/wrapped_gnn_markers_cpu.pt";

// -----------------------------------------------------------------------------

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir,
                     double& tend,
                     bool& run_time_vis,
                     bool& verbose,
                     bool& verbose_nn) {
    ChCLI cli(argv[0], "Polaris SPH terrain simulation");

    cli.AddOption<std::string>("", "terrain_dir", "Directory with terrain specification data");
    cli.AddOption<double>("", "tend", "Simulation end time [s]", std::to_string(tend));
    cli.AddOption<bool>("", "quiet", "Disable all messages during simulation");
    cli.AddOption<bool>("", "quiet_nn", "Disable all messages from NN model");
    cli.AddOption<bool>("", "run_time_vis", "Enable run-time visualization");

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

    run_time_vis = cli.GetAsType<bool>("run_time_vis");
    tend = cli.GetAsType<double>("tend");
    verbose = !cli.GetAsType<bool>("quiet");
    verbose_nn = !cli.GetAsType<bool>("quiet_nn");

    return true;
}

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
    bool Load(const std::string& pt_file);
    void Create(const std::string& terrain_dir, bool vis = true);
    void Synchronize(double time, const DriverInputs& driver_inputs);
    virtual void Advance(double step) override;

    double GetTimerDataIn() { return m_timer_data_in(); }
    double GetTimerModelEval() { return m_timer_model_eval(); }
    double GetTimerDataOut() { return m_timer_data_out(); }

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

    torch::jit::script::Module module;
    std::array<std::vector<ChVector<>>, 4> m_particle_forces;
    std::array<TerrainForce, 4> m_tire_forces;

    ChTimer<> m_timer_data_in;
    ChTimer<> m_timer_model_eval;
    ChTimer<> m_timer_data_out;

    bool m_verbose;
};

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

bool NNterrain::Load(const std::string& pt_file) {
    std::cout << "cuda version " << scatter::cuda_version() << std::endl;
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;

    std::ifstream is(pt_file, std::ios_base::binary);
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(is);
    } catch (const c10::Error& e) {
        cerr << "Load error: " << e.msg() << endl;
        return false;
    } catch (const std::exception& e) {
        cerr << "Load error other: " << e.what() << endl;
        return false;
    }
    cout << "Loaded model " << pt_file << endl;
    is.close();
    return true;
}

void NNterrain::Create(const std::string& terrain_dir, bool vis) {
    m_particles = chrono_types::make_shared<ChParticleCloud>();
    m_particles->SetFixed(true);

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

        if (num_particles > 1000000)
            break;
    }
    is.close();

    if (vis) {
        auto sph = chrono_types::make_shared<ChSphereShape>();
        sph->GetSphereGeometry().rad = 0.01;
        m_particles->AddVisualShape(sph);
    }

    m_sys.Add(m_particles);

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
    std::vector<torch::jit::IValue> inputs;

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

        // Do nothing if no particles under a wheel
        if (m_num_particles[i] == 0) {
            return;
        }

        // Load particle positions and velocities
        w_contact[i] = false;
        auto part_pos = torch::empty({(int)m_num_particles[i], 3}, torch::kFloat32);
        auto part_vel = torch::empty({(int)m_num_particles[i], 3}, torch::kFloat32);
        float* part_pos_data = part_pos.data<float>();
        float* part_vel_data = part_vel.data<float>();
        for (const auto& part : m_wheel_particles[i]) {
            ChVector<float> p(part->GetPos());
            *part_pos_data++ = p.x();
            *part_pos_data++ = p.y();
            *part_pos_data++ = p.z();
            ChVector<float> v(part->GetPos_dt());
            *part_vel_data++ = v.x();
            *part_vel_data++ = v.y();
            *part_vel_data++ = v.z();

            if (!w_contact[i] && (p - w_pos[i]).Length2() < tire_radius * tire_radius)
                w_contact[i] = true;
        }

        // Load wheel position, orientation, linear velocity, and angular velocity
        auto w_pos_t = torch::from_blob((void*)w_pos[i].data(), {3}, torch::kFloat32);
        auto w_rot_t = torch::from_blob((void*)w_rot[i].data(), {4}, torch::kFloat32);
        auto w_linvel_t = torch::from_blob((void*)w_linvel[i].data(), {3}, torch::kFloat32);
        auto w_angvel_t = torch::from_blob((void*)w_angvel[i].data(), {3}, torch::kFloat32);

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

        // Prepare the tuple input for this wheel
        std::vector<torch::jit::IValue> tuple;
        tuple.push_back(part_pos);
        tuple.push_back(part_vel);
        tuple.push_back(w_pos_t);
        tuple.push_back(w_rot_t);
        tuple.push_back(w_linvel_t);
        tuple.push_back(w_angvel_t);

        // Add this wheel's tuple to NN model inputs
        inputs.push_back(torch::ivalue::Tuple::create(tuple));
    }

    // Load vehicle data (1 tensor)
    auto drv_inputs = torch::tensor(
        {(float)driver_inputs.m_steering, (float)driver_inputs.m_throttle, (float)driver_inputs.m_braking},
        torch::kFloat32);
    inputs.push_back(drv_inputs);

    // Verbose flag
    inputs.push_back(m_verbose);

    m_timer_data_in.stop();

    // Invoke NN model

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

    m_timer_model_eval.start();
    torch::jit::IValue outputs;
    try {
        outputs = module.forward(inputs);
    } catch (const c10::Error& e) {
        cerr << "Execute error: " << e.msg() << endl;
        return;
    } catch (const std::exception& e) {
        cerr << "Execute error other: " << e.what() << endl;
        return;
    }
    m_timer_model_eval.stop();

    // Extract outputs

    m_timer_data_out.start();

    // Loop over all vehicle wheels
    for (int i = 0; i < 4; i++) {
        // Outputs for this wheel
        const auto& part_frc = outputs.toTuple()->elements()[i].toTensor();
        const auto& tire_frc = outputs.toTuple()->elements()[i + 4].toTensor();

        // Extract particle forces
        m_particle_forces[i].resize(m_num_particles[i]);
        for (size_t j = 0; j < m_num_particles[i]; j++) {
            m_particle_forces[i][j] =
                ChVector<>(part_frc[j][0].item<float>(), part_frc[j][1].item<float>(), part_frc[j][2].item<float>());
        }

        // Extract tire forces
        m_tire_forces[i].force =
            ChVector<>(tire_frc[0].item<float>(), tire_frc[1].item<float>(), tire_frc[2].item<float>());
        m_tire_forces[i].moment =
            ChVector<>(tire_frc[3].item<float>(), tire_frc[4].item<float>(), tire_frc[5].item<float>());
        m_tire_forces[i].point = ChVector<>(0, 0, 0);

        std::static_pointer_cast<ProxyTire>(m_wheels[i]->GetTire())->m_force = m_tire_forces[i];
        if (m_verbose) {
            std::cout << "  tire " << i << " force: " << m_tire_forces[i].force << std::endl;
        }
    }

    m_timer_data_out.stop();
}

void NNterrain::Advance(double step) {
    // Do nothing if there is at least one sampling box with no particles
    auto product = std::accumulate(m_num_particles.begin(), m_num_particles.end(), 1, std::multiplies<int>());
    if (product == 0)
        return;

    // Update state of particles in sampling boxes.
    // Assume mass=1 for all particles.
    double step2 = step * step / 2;
    for (int i = 0; i < 4; i++) {
        for (size_t j = 0; j < m_num_particles[i]; j++) {
            auto v = m_wheel_particles[i][j]->GetPos_dt() + m_particle_forces[i][j] * step;
            auto p = m_wheel_particles[i][j]->GetPos() + v * step + m_particle_forces[i][j] * step2;
            m_wheel_particles[i][j]->SetPos(p);
            m_wheel_particles[i][j]->SetPos_dt(v);
        }
    }
}

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
    // Parse command line arguments
    std::string terrain_dir;
    double tend = 30;
    bool run_time_vis = false;
    bool verbose = true;
    bool verbose_nn = true;

    if (!GetProblemSpecs(argc, argv, terrain_dir, tend, run_time_vis, verbose, verbose_nn)) {
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

    // Create driver
    auto path = ChBezierCurve::read(vehicle::GetDataFile(terrain_dir + "/path.txt"));
    double x_max = path->getPoint(path->getNumPoints() - 1).x() - 3.0;
    ChPathFollowerDriver driver(*vehicle, path, "my_path", target_speed);
    driver.GetSteeringController().SetLookAheadDistance(2.0);
    driver.GetSteeringController().SetGains(1.0, 0, 0);
    driver.GetSpeedController().SetGains(0.6, 0.05, 0);
    driver.Initialize();

    // Create terrain
    cout << "Create terrain..." << endl;
    NNterrain terrain(sys, vehicle);
    terrain.SetVerbose(verbose_nn);
    terrain.Create(terrain_dir);
    if (!terrain.Load(vehicle::GetDataFile(NN_module_name))) {
        return 1;
    }

#ifdef CHRONO_OPENGL
    opengl::ChVisualSystemOpenGL vis;
    if (run_time_vis) {
        vis.AttachSystem(&sys);
        vis.SetWindowTitle("Test");
        vis.SetWindowSize(1280, 720);
        vis.SetRenderMode(opengl::WIREFRAME);
        vis.Initialize();
        vis.SetCameraPosition(ChVector<>(-3, 0, 6), ChVector<>(5, 0, 0.5));
        vis.SetCameraVertical(CameraVerticalDir::Z);
    }
#endif

    // Simulation loop
    DriverInputs driver_inputs = {0, 0, 0};

    double step_size = 1e-3;
    double t = 0;
    int frame = 0;
    while (t < tend) {
#ifdef CHRONO_OPENGL
        if (run_time_vis) {
            if (vis.Run()) {
                vis.Render();
            } else {
                break;
            }
        }
#endif

        // Stop before end of patch
        if (vehicle->GetPos().x() > x_max)
            break;

        // Set current driver inputs
        driver_inputs = driver.GetInputs();

        if (t < 1) {
            driver_inputs.m_throttle = 0;
            driver_inputs.m_braking = 1;
        } else {
            ChClampValue(driver_inputs.m_throttle, driver_inputs.m_throttle, (t - 1) / 0.5);
        }

        if (verbose)
            cout << std::fixed << std::setprecision(3) << "t = " << t << "  STB = " << driver_inputs.m_steering << " "
                 << driver_inputs.m_throttle << " " << driver_inputs.m_braking << "  spd = " << vehicle->GetSpeed()
                 << "   timers = " << terrain.GetTimerDataIn() << " " << terrain.GetTimerModelEval() << " "
                 << terrain.GetTimerDataOut() << endl;

        // Synchronize subsystems
        driver.Synchronize(t);
        vehicle->Synchronize(t, driver_inputs, terrain);
        terrain.Synchronize(t, driver_inputs);

        // Advance system state
        driver.Advance(step_size);
        terrain.Advance(step_size);
        sys.DoStepDynamics(step_size);
        t += step_size;

        frame++;
    }

    return 0;
}
