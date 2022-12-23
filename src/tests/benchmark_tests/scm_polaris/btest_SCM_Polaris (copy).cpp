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
#include "chrono/utils/ChBenchmark.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/ChDriver.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"


#include "chrono_models/vehicle/hmmwv/HMMWV.h"

#include "chrono_thirdparty/filesystem/path.h"

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
double step_size = 1e-3;

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

template <int N>
class SCMPolarisTest : public utils::ChBenchmarkTest {
  public:
    SCMPolarisTest();
    ~SCMPolarisTest() { delete m_system; }

    ChSystem* GetSystem() override { return m_system; }
    void ExecuteStep() override { m_system->DoStepDynamics(m_step); }

    // void SimulateVis();

  private:
    ChSystem* m_system;
    double m_length;
    double m_step;
};

template <int N>
SCMPolarisTest<N>::SCMPolarisTest() : m_length(0.25), m_step(1e-3) {
    // Parse command line arguments
    // bool verbose = true;
    // bool wheel_output = true;      // save individual wheel output files
    // double output_major_fps = 50;
    // std::cout<<"Hhh"<<std::endl;
    // // --------------------
    // // Create the Chrono systems
    // // --------------------
    m_system= new ChSystemNSC;
    m_system->SetNumThreads(std::min(8, ChOMP::GetNumProcs()));
    const ChVector<> gravity(0, 0, -9.81);
    m_system->Set_G_acc(gravity);

    // --------------------
    // Create the Polaris vehicle
    // --------------------
    cout << "Create vehicle..." << endl;
    auto vehicle = CreateVehicle(model, *m_system, init_pos);

    // ------------------
    // Create the terrain
    // ------------------
    // SCMDeformableTerrain terrain(system);
    SCMDeformableTerrain terrain(m_system);
    terrain.SetSoilParameters(2e6,   // Bekker Kphi
                                0,     // Bekker Kc
                                1.1,   // Bekker n exponent
                                0,     // Mohr cohesive limit (Pa)
                                30,    // Mohr friction limit (degrees)
                                0.01,  // Janosi shear coefficient (m)
                                2e8,   // Elastic stiffness (Pa/m), before plastic yield
                                3e4    // Damping (Pa s/m), proportional to negative vertical speed (optional)
    );

    terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_SINKAGE, 0, 0.1);
    terrain.Initialize(terrainLength, terrainWidth, delta);
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
}

// template <int N>
// void SCMPolarisTest<N>::SimulateVis() {
// #ifdef CHRONO_IRRLICHT
//     double offset = N * m_length;

//     // Create the Irrlicht visualization system
//     auto vis = chrono_types::make_shared<irrlicht::ChVisualSystemIrrlicht>();
//     vis->AttachSystem(m_system);
//     vis->SetWindowSize(800, 600);
//     vis->SetWindowTitle("Pendulum chain");
//     vis->Initialize();
//     vis->AddLogo();
//     vis->AddSkyBox();
//     vis->AddTypicalLights();
//     vis->AddCamera(ChVector<>(0, -offset / 2, offset), ChVector<>(0, -offset / 2, 0));

//     while (vis->Run()) {
//         vis->BeginScene();
//         vis->Render();
//         m_system->DoStepDynamics(m_step);
//         vis->EndScene();
//     }
// #endif
// }

// =============================================================================

#define NUM_SKIP_STEPS 2000  // number of steps for hot start
#define NUM_SIM_STEPS 1000   // number of simulation steps for each benchmark

CH_BM_SIMULATION_LOOP(SCMPolaris04, SCMPolarisTest<4>,  NUM_SKIP_STEPS, NUM_SIM_STEPS, 5);


// =============================================================================

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

// #ifdef CHRONO_IRRLICHT
//     if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
//         SCMPolarisTest<4> test;
//         test.SimulateVis();
//         return 0;
//     }
// #endif

    ::benchmark::RunSpecifiedBenchmarks();
}
