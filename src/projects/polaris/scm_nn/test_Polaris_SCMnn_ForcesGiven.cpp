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
#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"

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

class CustomTerrain : public ChTerrain {
  public:
    CustomTerrain(ChSystem& sys, WheeledVehicle& vehicle);
    bool Load(const std::string& pt_file);
    void Create(const std::string& terrain_dir, bool vis = true);
    void Synchronize(double time, int frame);
    virtual void Advance(double step) override;

    void SetVerbose(bool val) { m_verbose = val; }

  private:
    ChSystem& m_sys;
    const WheeledVehicle& m_vehicle;
    std::array<std::shared_ptr<ChWheel>, 4> m_wheels;
    std::shared_ptr<ChParticleCloud> m_particles;
    
    ChVector<> m_box_size;
    ChVector<> m_box_offset;
    std::array<std::vector<ChAparticle*>, 4> m_wheel_particles;
    std::array<size_t, 4> m_num_particles;
    double m_mbs_inputs[10001][90];

    torch::jit::script::Module module;
    std::array<std::vector<ChVector<>>, 4> m_particle_displacements;
    std::array<TerrainForce, 4> m_tire_forces;

    bool m_verbose;
};

// -----------------------------------------------------------------------------

CustomTerrain::CustomTerrain(ChSystem& sys, WheeledVehicle& vehicle) : m_sys(sys), m_vehicle(vehicle), m_verbose(true)  {
    m_wheels[0] = vehicle.GetWheel(0, LEFT);
    m_wheels[1] = vehicle.GetWheel(0, RIGHT);
    m_wheels[2] = vehicle.GetWheel(1, LEFT);
    m_wheels[3] = vehicle.GetWheel(1, RIGHT);

    // Set default size and offset of sampling box
    double tire_radius = m_wheels[0]->GetTire()->GetRadius();
    double tire_width = m_wheels[0]->GetTire()->GetWidth();
    m_box_size.x() = 2.0 * std::sqrt(3.0) * tire_radius;
    m_box_size.y() = 1.5 * tire_width;
    m_box_size.z() = 0.2;
    m_box_offset = ChVector<>(0.0, 0.0, 0.0);
}


bool CustomTerrain::Load(const std::string& pt_file) {
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


void CustomTerrain::Create(const std::string& terrain_dir, bool vis) {
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

        if (num_particles > 1000000)
            break;
    }
    is.close();


    int counter=0;
    std::string line2;
    std::string cell2;
    std::ifstream is2(vehicle::GetDataFile(terrain_dir + "/mbs.txt"));
    while (getline(is2, line2)) {
        std::stringstream ls2(line2);
        for (int i = 0; i < 90; i++) {
            getline(ls2, cell2, ',');
            m_mbs_inputs[counter][i]  = stod(cell2);
            
        }
        counter++;
    }
    is2.close();


    // for (int i = 0; i < 1000; i++) {
    //     for (int j = 0; j < 91; j++) {
    //         std::cout<<m_mbs_inputs[i][j]<<",";
    //     }
    //     std::cout<<"done"<<std::endl;
    // }


    m_sys.Add(m_particles);

    if (vis) {
        auto sph = chrono_types::make_shared<ChSphereShape>();
        sph->GetSphereGeometry().rad = 0.01;
        m_particles->AddVisualShape(sph);
    }

    

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


void CustomTerrain::Synchronize(double time,int frame) {
    // Prepare NN model inputs
    const auto& p_all = m_particles->GetParticles();
    std::vector<torch::jit::IValue> inputs;

    // Loop over all vehicle wheels
    std::array<ChVector<float>, 4> w_pos;
    std::array<ChQuaternion<float>, 4> w_rot;
    std::array<ChVector<float>, 4> w_nrm;
    std::array<ChVector<float>, 4> w_linvel;
    std::array<ChVector<float>, 4> w_angvel;
    std::array<bool, 4> w_contact;
    for (int i = 0; i < 4; i++) {
        

        // Call the forces and torques
        double Fx = m_mbs_inputs[frame][66+6*i];
        double Fy = m_mbs_inputs[frame][66+6*i+1];
        double Fz = m_mbs_inputs[frame][66+6*i+2];
        double Mx = m_mbs_inputs[frame][66+6*i+3];
        double My = m_mbs_inputs[frame][66+6*i+4];
        double Mz = m_mbs_inputs[frame][66+6*i+5];

   
        // Tire force and moment in tire frame
        ChVector<> tire_F(Fx, Fy, Fz);
        ChVector<> tire_M(Mx, My, Mz);

        // Load the tire force structure (all expressed in absolute frame)
        m_tire_forces[i].force = tire_F;
        m_tire_forces[i].moment = tire_M;
        m_tire_forces[i].point = (0, 0, 0);
    }

    // 2. Add tire forces as external forces to spindle bodies

    for (int i = 0; i < 4; i++) {
        m_wheels[i]->GetSpindle()->Accumulate_force(m_tire_forces[i].force, m_tire_forces[i].point, false);
        m_wheels[i]->GetSpindle()->Accumulate_torque(m_tire_forces[i].moment, false);
    }

}

void CustomTerrain::Advance(double step) {
    // // Do nothing if there is at least one sampling box with no particles
    // auto product = std::accumulate(m_num_particles.begin(), m_num_particles.end(), 1, std::multiplies<int>());
    // if (product == 0)
    //     return;

    // // Update state of particles in sampling boxes.
    // // Assume mass=1 for all particles.
    // double step2 = step * step / 2;
    // for (int i = 0; i < 4; i++) {
    //     for (size_t j = 0; j < m_num_particles[i]; j++) {
    //         auto p_disp = m_particle_displacements[i][j];
    //         auto p_current = m_wheel_particles[i][j]->GetPos();
    //         auto p_actual = p_current+p_disp;
    //         m_wheel_particles[i][j]->SetPos(p_actual);
    //     }
    // }
}

// -----------------------------------------------------------------------------

std::shared_ptr<WheeledVehicle> CreateVehicle(ChSystem& sys, const ChCoordsys<>& init_pos) {
    
    std::string vehicle_json = "Polaris/Polaris.json";
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
    std::string terrain_dir = "terrain/scm/testterrainnn";
    double tend = 20000.0;
    bool run_time_vis = true;
    bool verbose = true;
    bool verbose_nn = true;



    // Create the Chrono systems
    ChSystemNSC sys;

    sys.Set_G_acc(ChVector<>(0, 0, -9.81));

    // Create vehicle
    ChCoordsys<> init_pos(ChVector<>(1.3, 0, 0.1), QUNIT);

    std::string vehicle_json = "Polaris/Polaris.json";
    std::string powertrain_json = "Polaris/Polaris_SimpleMapPowertrain.json";
    std::string tire_json = "Polaris/Polaris_RigidTire.json";

    // Create and initialize the vehicle
    WheeledVehicle vehicle(&sys, vehicle::GetDataFile(vehicle_json));
    vehicle.Initialize(init_pos);
    vehicle.GetChassis()->SetFixed(false);
    vehicle.SetChassisVisualizationType(VisualizationType::MESH);
    vehicle.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(powertrain_json));
    vehicle.InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : vehicle.GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(tire_json));
            vehicle.InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }


    // Create terrain
    cout << "Create terrain..." << endl;
    CustomTerrain terrain(sys, vehicle);
    terrain.SetVerbose(verbose_nn);
    terrain.Create(terrain_dir);
    if (!terrain.Load(vehicle::GetDataFile(NN_module_name))) {
        return 1;
    }

// Create Irrilicht visualization
    ChWheeledVehicleVisualSystemIrrlicht vis;
    vis.SetWindowTitle("Polaris - Custom terrain example");
    vis.SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), 5.0, 0.5);
    vis.Initialize();
    vis.AddTypicalLights();
    vis.AddSkyBox();
    vis.AddLogo();
    vis.AttachVehicle(&vehicle);


    // Simulation loop
    DriverInputs driver_inputs = {0.0, 0.0, 1.0};

    double step_size = 1e-3;
    double t = 0;
    int frame = 0;
    while (t < tend) {
        if (run_time_vis) {
         vis.Run();
         vis.BeginScene();
         vis.Render();
         vis.EndScene();
        }

        // driver_inputs.m_steering = 0.0;
        // driver_inputs.m_throttle = 0.0;
        // driver_inputs.m_braking = 1.0;


        // Synchronize subsystems
        vehicle.Synchronize(t, driver_inputs, terrain);
        terrain.Synchronize(t,frame);

        // Advance system state
        terrain.Advance(step_size);
        vis.Advance(step_size);
        sys.DoStepDynamics(step_size);
        t += step_size;

        frame++;
    }

    return 0;
}