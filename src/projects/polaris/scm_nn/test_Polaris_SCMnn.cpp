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


// // -----------------------------------------------------------------------------

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
    void Create(const std::string& terrain_dir, bool vis = true);
    void Synchronize(double time, const DriverInputs& driver_inputs);
    virtual void Advance(double step) override;

  private:
    ChSystem& m_sys;
    std::shared_ptr<WheeledVehicle> m_vehicle;
    std::array<std::shared_ptr<ChWheel>, 4> m_wheels;
    std::array<TerrainForce, 4> m_tire_forces;
    bool m_verbose;
};


NNterrain::NNterrain(ChSystem& sys, std::shared_ptr<WheeledVehicle> vehicle)
    : m_sys(sys), m_vehicle(vehicle), m_verbose(true) {
    m_wheels[0] = vehicle->GetWheel(0, LEFT);
    m_wheels[1] = vehicle->GetWheel(0, RIGHT);
    m_wheels[2] = vehicle->GetWheel(1, LEFT);
    m_wheels[3] = vehicle->GetWheel(1, RIGHT);
}

void NNterrain::Synchronize(double time, const DriverInputs& driver_inputs) {


    // Loop over all vehicle wheels
    for (int i = 0; i < 4; i++) {
        // Extract tire forces
        m_tire_forces[i].force =
            ChVector<>(0.0,0.0,1000.0);
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

    std::string terrain_dir="terrain/sph/testterrainnn";
    double tend = 30;
    bool run_time_vis = true;

    // Create the Chrono systems
    ChSystemNSC sys;
    sys.Set_G_acc(ChVector<>(0, 0, -9.81));

    // Create vehicle
    ChCoordsys<> init_pos(ChVector<>(4, 0, 0.1), Q_from_AngX(0.0) * Q_from_AngY(0.0));
    auto vehicle = CreateVehicle(sys, init_pos);


    // Create terrain
    cout << "Create terrain..." << endl;
    NNterrain terrain(sys, vehicle);


    opengl::ChVisualSystemOpenGL vis;
    vis.AttachSystem(&sys);
    vis.SetWindowTitle("test_Polaris_SCMnn");
    vis.SetWindowSize(1600, 900);
    vis.SetRenderMode(opengl::WIREFRAME);
    vis.SetParticleRenderMode(0.05f, opengl::POINTS);
    vis.Initialize();
    vis.AddCamera(ChVector<>(-3.0, 0.0, 6.0), ChVector<>(5.0, 0.0, 0.5));
    vis.SetCameraVertical(CameraVerticalDir::Z);


    // Simulation loop
    DriverInputs driver_inputs = {0.0, 0.0, 0.0};

    double step_size = 1e-3;
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

        driver_inputs.m_steering = 0.0;
        driver_inputs.m_throttle = 0.0;
        driver_inputs.m_braking = 1.0;

        // Synchronize subsystems
        vehicle->Synchronize(t, driver_inputs, terrain);
        terrain.Synchronize(t, driver_inputs);

        // Advance system state
        terrain.Advance(step_size);
        sys.DoStepDynamics(step_size);
        t += step_size;

        frame++;
    }

    return 0;
}
