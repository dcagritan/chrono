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

// #ifdef CHRONO_OPENGL
//     #include "chrono_opengl/ChVisualSystemOpenGL.h"
// #endif
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

// NN model
// std::string NN_module_name = "terrain/scm/wrapped_gnn_cpu_settling_flat.pt";
// std::string NN_module_name = "terrain/scm/wrapped_gnn_cpu_settling_flat_stepsize2_clamp_sinkage.pt";
std::string NN_module_name = "terrain/scm/wrapped_gnn_cpu_rol_20.pt";

// -----------------------------------------------------------------------------

class CustomTerrain : public ChTerrain {
  public:
    CustomTerrain(ChSystem& sys, WheeledVehicle& vehicle);
    bool Load(const std::string& pt_file);
    void Create(const std::string& terrain_dir, bool vis = true);
    void Synchronize(double time, int frame, bool debug_output);
    virtual void Advance(double step) override;
    void SetVerbose(bool val) { m_verbose = val; }
    void WriteVertices(double time);

  private:
    ChSystem& m_sys;
    const WheeledVehicle& m_vehicle;
    std::array<std::shared_ptr<ChWheel>, 4> m_wheels;
    std::shared_ptr<ChParticleCloud> m_particles;
    
    ChVector<> m_box_size;
    ChVector<> m_box_offset;
    std::array<std::vector<ChAparticle*>, 4> m_wheel_particles;
    std::array<size_t, 4> m_num_particles;

    torch::jit::script::Module module;
    std::array<std::vector<ChVector<>>, 4> m_particle_displacements;
    std::array<TerrainForce, 4> m_tire_forces;
    bool m_verbose;

};

// -----------------------------------------------------------------------------

CustomTerrain::CustomTerrain(ChSystem& sys, WheeledVehicle& vehicle) : m_sys(sys), m_vehicle(vehicle)  {
    m_wheels[0] = vehicle.GetWheel(0, LEFT);
    m_wheels[1] = vehicle.GetWheel(0, RIGHT);
    m_wheels[2] = vehicle.GetWheel(1, LEFT);
    m_wheels[3] = vehicle.GetWheel(1, RIGHT);

    // Set default size and offset of sampling box
    double tire_radius = m_wheels[0]->GetTire()->GetRadius();
    double tire_width = m_wheels[0]->GetTire()->GetWidth();
    m_box_size.x() = 2.0 * std::sqrt(3.0) * tire_radius;
    m_box_size.y() = 1.5 * tire_width;
    m_box_size.z() = 2.2;
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


void CustomTerrain::Synchronize(double time,int frame, bool debug_output) {

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
        auto part_pos = torch::empty({(int)m_num_particles[i], 4}, torch::kFloat32);
        float* part_pos_data = part_pos.data<float>();
        for (const auto& part : m_wheel_particles[i]) {
            ChVector<float> p(part->GetPos());
            *part_pos_data++ = p.x();
            *part_pos_data++ = p.y();
            *part_pos_data++ = p.z();
            *part_pos_data++ = -p.z();

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

#if 0
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
        tuple.push_back(w_pos_t);
        tuple.push_back(w_rot_t);
        tuple.push_back(w_linvel_t);
        tuple.push_back(w_angvel_t);

        // Add this wheel's tuple to NN model inputs
        inputs.push_back(torch::ivalue::Tuple::create(tuple));

    }

        // Invoke NN model

#if 0
    for (int i = 0; i < 4; i++) {
    std::cout << "  inputs[i]:                                            "           //
                << &inputs[i] << std::endl;                                             //
    std::cout << "  inputs[i].toTuple()->elements()[2].toTensor():        "           //
                << &inputs[i].toTuple()->elements()[2].toTensor() << std::endl;         //
    std::cout << "  inputs[i].toTuple()->elements()[2].toTensor().data(): "           //
                << &inputs[i].toTuple()->elements()[2].toTensor().data() << std::endl;  //
    const auto& wpt = inputs[i].toTuple()->elements()[2].toTensor();                  //
    std::cout << "  wheel " << i << "  position tensor: "                             //
                << wpt[0].item<float>() << " " << wpt[1].item<float>() << " " << wpt[2].item<float>() << std::endl;
    }
#endif

    // m_timer_model_eval.start();
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

    // Loop over all vehicle wheels
    for (int i = 0; i < 4; i++) {
        // Outputs for this wheel
        const auto& part_disp = outputs.toTuple()->elements()[i].toTensor();
        const auto& tire_frc = outputs.toTuple()->elements()[i + 4].toTensor();

        ChVector<> disc_center = m_wheels[i]->GetPos();

        // Extract particle displacements
        m_particle_displacements[i].resize(m_num_particles[i]);
        for (size_t j = 0; j < m_num_particles[i]; j++) {
            m_particle_displacements[i][j] =
                ChVector<>(part_disp[j][0].item<float>(), part_disp[j][1].item<float>(), part_disp[j][2].item<float>());
        }

        // Extract tire forces
        m_tire_forces[i].force =
            ChVector<>(tire_frc[0].item<float>(), tire_frc[1].item<float>(), tire_frc[2].item<float>());
        m_tire_forces[i].moment =
            ChVector<>(tire_frc[3].item<float>(), tire_frc[4].item<float>(), tire_frc[5].item<float>());
        m_tire_forces[i].point = disc_center;

        m_wheels[i]->GetSpindle()->Accumulate_force(m_tire_forces[i].force, m_tire_forces[i].point, false);
        m_wheels[i]->GetSpindle()->Accumulate_torque(m_tire_forces[i].moment, false);
        if (m_verbose) {
            std::cout << "time= "<<time<<"  tire " << i << " force: " << m_tire_forces[i].force << std::endl;
        }
    }    
}

void CustomTerrain::Advance(double step) {
    // Do nothing if there is at least one sampling box with no particles
    // auto product = std::accumulate(m_num_particles.begin(), m_num_particles.end(), 1, std::multiplies<int>());
    // if (product == 0)
    //     return;
    // Update state of particles in sampling boxes.
    // double maxdisplacement=0.0;
    for (int i = 0; i < 4; i++) {
        for (size_t j = 0; j < m_num_particles[i]; j++) {
            auto p_disp = m_particle_displacements[i][j];
            // std::cout<<"p_disp= "<<p_disp<<std::endl;
            // if (p_disp.z()<maxdisplacement)
            //    maxdisplacement= p_disp.z();
            // std::cout<<"zdisp.z()"<<p_disp.z()<<std::endl;
            auto p_current = m_wheel_particles[i][j]->GetPos();
            // if (p_current.z()<=-0.12)
            //     p_disp.z()=0.0;
            auto p_actual = p_current+p_disp;
            m_wheel_particles[i][j]->SetPos(p_actual);
        }
        // std::cout<<"Max displacement of the box "<<i<<"is= "<<maxdisplacement<<std::endl;
    }
            // std::cout<<" Max displacement of all the boxes is "<<maxdisplacement<<std::endl;
}

void CustomTerrain::WriteVertices(double time) {
    const auto& p_all = m_particles->GetParticles();
    std::string filename = "disp/displacement_" + std::to_string(time) + "_" +  ".csv";
    std::ofstream stream;
    stream.open(filename, std::ios_base::trunc);

    // for (int i = 0; i < 4; i++) {
    //     for (size_t j = 0; j < m_num_particles[i]; j++) {
    //         auto p_current = m_wheel_particles[i][j]->GetPos();
    //         stream << p_current << "\n";
    //     }
    //     // std::cout<<"Max displacement of the box "<<i<<"is= "<<maxdisplacement<<std::endl;
    // }
  


    for (const auto& part : p_all) {
            ChVector<float> p(part->GetPos());
            // std::cout<<p<<std::endl;
            stream << p << "\n";
        }

    // size_t start = 0;

    // // Vehicle position, orientation, linear and angular velocities
    // for (int j = 0; j < 13; j++)
    //     stream << o[start + j] << ", ";
    // stream << "\n";
    // start += 13;

    // // Wheel position, orientation, linear and angular velocities
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 13; j++)
    //         stream << o[start + j] << ", ";
    //     stream << "\n";
    //     start += 13;
    // }

    // // Tire force and moment
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 6; j++)
    //         stream << o[start + j] << ", ";
    //     stream << "\n";
    //     start += 6;
    // }

    stream.close();
}

// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse arguments
    std::string terrain_dir = "terrain/scm/testterrain";
    double tend = 1.0;
    bool run_time_vis = true;
    bool debug_output = false;
    bool verbose_nn = true;

    // Create the Chrono systems
    ChSystemNSC sys;
    sys.Set_G_acc(ChVector<>(0, 0, -9.81));


    // Create and initialize the vehicle
    ChCoordsys<> init_pos(ChVector<>(1.3, 0, 0.1), QUNIT);
    std::string vehicle_json = "Polaris/Polaris.json";
    std::string powertrain_json = "Polaris/Polaris_SimpleMapPowertrain.json";
    std::string tire_json = "Polaris/Polaris_RigidTire.json";
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
    terrain.Create(terrain_dir,true);
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

    double step_size = 2e-3;
    double t = 0;
    int frame = 0;
    while (t < tend) {

        if (run_time_vis) {
         vis.Run();
         vis.BeginScene();
         vis.Render();
         vis.EndScene();
        }

        DriverInputs driver_inputs = {0.0, 0.0, 0.0};

        // Synchronize subsystems
        vehicle.Synchronize(t, driver_inputs, terrain);
        terrain.Synchronize(t,frame,debug_output);

        // Advance system state
        // std::cout<<"time "<<t;
        terrain.Advance(step_size);
        // terrain.WriteVertices(t);
        vis.Advance(step_size);
        sys.DoStepDynamics(step_size);
        t += step_size;

        frame++;
    }

    return 0;
}