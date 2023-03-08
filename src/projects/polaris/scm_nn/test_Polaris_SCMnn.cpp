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

enum class MRZR_MODEL { ORIGINAL, MODIFIED };

MRZR_MODEL model = MRZR_MODEL::MODIFIED;

// Speed controller target speed (in m/s)
double target_speed = 7;

// NN model
std::string NN_module_name = "terrain/scm/wrapped_gnn_markers_cpu.pt";


// -----------------------------------------------------------------------------

class ProxyTire : public RigidTire {
  public:
    ProxyTire(const std::string& filename) : RigidTire(filename) {}
    ProxyTire(const rapidjson::Document& d) : RigidTire(d) {}
    virtual TerrainForce ReportTireForce(ChTerrain* terrain) const override { return m_force; }
    virtual TerrainForce GetTireForce() const { return m_force; }
    TerrainForce m_force;
};

// -----------------------------------------------------------------------------

class NNterrain : public ChTerrain {
  public:
    NNterrain(ChSystem& sys, std::shared_ptr<WheeledVehicle> vehicle);
    bool Load(const std::string& pt_file);
    void Create(const std::string& terrain_dir, bool vis = true);
    void Synchronize(double time);
    virtual void Advance(double step) override;

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

    // // Create the random particles
    // for (int np = 0; np < 100; ++np)
    //     m_particles->AddParticle(ChCoordsys<>(ChVector<>(ChRandom() - 2, 0.5, ChRandom() + 2)));

    int num_particles = 0;
    ChVector<> marker;
    std::string line;
    std::string cell;

    std::ifstream is(vehicle::GetDataFile(terrain_dir + "/vertices_0.txt"));
    getline(is, line);  // Comment line
    while (getline(is, line)) {
        std::stringstream ls(line);
        for (int i = 0; i < 3; i++) {
            getline(ls, cell, ',');
            marker[i] = stod(cell);
        }
        m_particles->AddParticle(ChCoordsys<>(marker));
        num_particles++;

        if (num_particles > 100000)
            break;
    }
    is.close();


    m_sys.Add(m_particles);

    if (vis) {
        auto sph = chrono_types::make_shared<ChSphereShape>();
        sph->GetSphereGeometry().rad = 0.01;
        m_particles->AddVisualShape(sph);
    }

    

    // // Initial size of sampling box particle vectors
    // for (int i = 0; i < 4; i++)
    //     m_wheel_particles[i].resize(num_particles);
}


void NNterrain::Synchronize(double time) {


    // Loop over all vehicle wheels
    for (int i = 0; i < 4; i++) {

        // Extract tire forces
        m_tire_forces[i].force =
            ChVector<>(0.0,0.0,0.0);
        m_tire_forces[i].moment =
            ChVector<>(0.0,0.0,0.0);
        m_tire_forces[i].point = ChVector<>(0, 0, 0);

        std::static_pointer_cast<ProxyTire>(m_wheels[i]->GetTire())->m_force = m_tire_forces[i];
        if (true) {
            std::cout << "  tire " << i << " force: " << m_tire_forces[i].force << std::endl;
        }
    }

}

void NNterrain::Advance(double step) {
}

// -----------------------------------------------------------------------------

std::shared_ptr<WheeledVehicle> CreateVehicle(ChSystem& sys, const ChCoordsys<>& init_pos) {
    std::string model_dir = (model == MRZR_MODEL::ORIGINAL) ? "mrzr/JSON_orig/" : "mrzr/JSON_new/";

    std::string vehicle_json = model_dir + "vehicle/MRZR.json";
    ////std::string powertrain_json = model_dir + "powertrain/MRZR_SimplePowertrain.json";
    std::string powertrain_json = model_dir + "powertrain/MRZR_SimpleMapPowertrain.json";
    std::string tire_json = model_dir + "tire/MRZR_RigidTire.json";

    std::string tire_coll_obj = "mrzr/meshes_new/Polaris_tire_collision.obj";

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
            auto tire = chrono_types::make_shared<ProxyTire>(vehicle::GetDataFile(tire_json));
            vehicle->InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    return vehicle;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string terrain_dir = "terrain/scm/testterrainnn";
    double tend = 30;
    bool run_time_vis = true;
    bool verbose = true;
    bool verbose_nn = true;

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
    ChCoordsys<> init_pos(ChVector<>(4, 0, 0.02 + 4 * std::sin(slope)), Q_from_AngX(banking) * Q_from_AngY(-slope));
    auto vehicle = CreateVehicle(sys, init_pos);


    // Create terrain
    cout << "Create terrain..." << endl;
    NNterrain terrain(sys, vehicle);
    terrain.SetVerbose(verbose_nn);
    terrain.Create(terrain_dir);
    // if (!terrain.Load(vehicle::GetDataFile(NN_module_name))) {
    //     return 1;
    // }

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
                // vis.Render();
                vis.BeginScene();
                vis.Render();
                vis.EndScene();
            } else {
                break;
            }
        }
#endif

        driver_inputs.m_steering = 0.0;
        driver_inputs.m_throttle = 0.0;
        driver_inputs.m_braking = 1.0;


        // Synchronize subsystems
        vehicle->Synchronize(t, driver_inputs, terrain);
        terrain.Synchronize(t);

        // Advance system state
        terrain.Advance(step_size);
        sys.DoStepDynamics(step_size);
        t += step_size;

        frame++;
    }

    return 0;
}
