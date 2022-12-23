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


#include <cstdio>
#include <cmath>
#include <vector>

#include <iostream>
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/core/ChTimer.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/utils/ChBenchmark.h"
#include <benchmark/benchmark.h>

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/ChDriver.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"


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

// Benchmarking fixture: create system and add bodies
class SCMPolarisFixture : public ::benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& st) override {
        // Parse command line arguments

    current_time=0.0;
    time_step=step_size;
    bool verbose = true;
    bool wheel_output = true;      // save individual wheel output files
    // --------------------
    // Create the Chrono systems
    // --------------------
    m_system = new ChSystemNSC;
    m_system->SetNumThreads(std::min(8, ChOMP::GetNumProcs()));
    const ChVector<> gravity(0, 0, -9.81);
    m_system->Set_G_acc(gravity);

    // --------------------
    // Create the Polaris vehicle
    // --------------------
    // cout << "Create vehicle..." << endl;
    // auto vehicle = CreateVehicle(model, m_system, init_pos);

    std::string model_dir = (model == PolarisModel::ORIGINAL) ? "mrzr/JSON_orig/" : "mrzr/JSON_new/";

    std::string vehicle_json = model_dir + "vehicle/MRZR.json";
    ////std::string powertrain_json = model_dir + "powertrain/MRZR_SimplePowertrain.json";
    std::string powertrain_json = model_dir + "powertrain/MRZR_SimpleMapPowertrain.json";
    std::string tire_json = model_dir + "tire/MRZR_RigidTire.json";

    // Create and initialize the vehicle
    m_vehicle = new WheeledVehicle(m_system, vehicle::GetDataFile(vehicle_json));
    // auto vehicle = chrono_types::make_shared<WheeledVehicle>(m_system, vehicle::GetDataFile(vehicle_json));
    m_vehicle->Initialize(init_pos);
    m_vehicle->GetChassis()->SetFixed(false);
    m_vehicle->SetChassisVisualizationType(VisualizationType::MESH);
    m_vehicle->SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    m_vehicle->SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    m_vehicle->SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(powertrain_json));
    m_vehicle->InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : m_vehicle->GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(tire_json));
            m_vehicle->InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    // ------------------
    // Create the terrain
    // ------------------
    m_terrain = new SCMDeformableTerrain(m_system);
    // SCMDeformableTerrain terrain(m_system);
    m_terrain->SetSoilParameters(2e6,   // Bekker Kphi
                                0,     // Bekker Kc
                                1.1,   // Bekker n exponent
                                0,     // Mohr cohesive limit (Pa)
                                30,    // Mohr friction limit (degrees)
                                0.01,  // Janosi shear coefficient (m)
                                2e8,   // Elastic stiffness (Pa/m), before plastic yield
                                3e4    // Damping (Pa s/m), proportional to negative vertical speed (optional)
    );

    m_terrain->SetPlotType(vehicle::SCMDeformableTerrain::PLOT_SINKAGE, 0, 0.1);
    m_terrain->Initialize(terrainLength, terrainWidth, delta);
    // auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    m_vis = new ChWheeledVehicleVisualSystemIrrlicht;
    m_vis->SetWindowTitle("HMMWV Deformable Soil Demo");
    m_vis->SetChaseCamera(trackPoint, 6.0, 0.5);
    m_vis->Initialize();
    m_vis->AddLightDirectional();
    m_vis->AddSkyBox();
    m_vis->AddLogo();
    // ChVehicle* castedvehicle = m_vehicle.get();
    // auto castedvehicle = std::static_pointer_cast<ChVehicle>(vehicle);
    // ChVehicle* castedvehicle = (ChVehicle*)vehicle
    m_vis->AttachVehicle(m_vehicle);

    // --------------------
    // Create driver system
    // --------------------
    // MyDriver driver(*castedvehicle, 0.5);
    // driver.Initialize();

    m_driver = new MyDriver(*m_vehicle, 0.5);
    m_driver->Initialize();;


    // ---------------
    // Simulation loop
    // ---------------
    std::cout << "Total vehicle mass: " << m_vehicle->GetMass() << std::endl;

    // Solver settings.
    m_system->SetSolverMaxIterations(50);


    // Initialize simulation frame counter
    int step_number = 0;

    // // double time = m_system->GetChTime();
     for (step_number;step_number<2000;step_number++)
     { 
        // // Render scene
        // m_vis->BeginScene();
        // m_vis->Render();
        // tools::drawColorbar(m_vis, 0, 0.1, "Sinkage", 30);
        // m_vis->EndScene();

        // Driver inputs
        m_driver_inputs = m_driver->GetInputs();

        // // Update modules
        m_driver->Synchronize(current_time);
        m_terrain->Synchronize(current_time);
        m_vehicle->Synchronize(current_time, m_driver_inputs, *m_terrain);
        m_vis->Synchronize("", m_driver_inputs);

        // Advance dynamics
        m_system->DoStepDynamics(time_step);
        m_vis->Advance(step_size);

        // Increment time
        current_time+=time_step;
     }
     std::cout<<"current_time= "<<current_time<<std::endl;

    }

    void TearDown(const ::benchmark::State&) override {
        delete m_system;
        delete m_driver;
        delete m_vehicle;
        delete m_terrain;
        delete m_vis;
    }

public:
    double current_time;
    double time_step;
    ChSystem* m_system;
    MyDriver* m_driver;
    WheeledVehicle* m_vehicle;
    SCMDeformableTerrain* m_terrain;
    ChVehicleVisualSystemIrrlicht* m_vis;
    DriverInputs m_driver_inputs;
    
};

// Utility macros for benchmarking body operations with different signatures
// #define BM_BODY_OP_TIME(OP)                                                 \
//     BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {          \
//         m_system->OP(1e-3);   \
//         st.SetItemsProcessed(st.iterations());                 \
//     }                                                                       \
//     BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMicrosecond);


#define BM_DRIVER_OP_VOID(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_driver->GetInputs();                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);    
    // BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond);  


#define BM_DRIVER_SYNCRONIZE_TIME(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_driver->Synchronize(current_time);                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);     


#define BM_TERRAIN_SYNCRONIZE_TIME(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_terrain->Synchronize(current_time);                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);    

#define BM_VEHICLE_SYNCRONIZE_TIME(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_vehicle->Synchronize(current_time, m_driver_inputs, *m_terrain);                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);               

    

#define BM_DRIVER_OP_TIME(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_driver->OP(current_time);                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);       

#define BM_SYSTEM_OP_TIMESTEP(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_system->OP(1e-3);                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);      
    // BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond);    


#define BM_VIS_OP_TIMESTEP(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_vis->Advance(1e-3);                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);      
    // BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond);        

#define BM_DRIVER_OP_TIME(OP)                                                 \
    BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
        for (auto _ : st) {                                                 \
                m_driver->OP(current_time);                                         \
        }                                                                   \
        st.SetItemsProcessed(st.iterations()); \
    }                                                                       \
    BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(1);    

// #define BM_BODY_OP_TIMESTEP(OP)                                                 \
//     BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
//         for (auto _ : st) {                                                 \
//                 this->SayHello();                                         \
//         }                                                                   \
//         st.SetItemsProcessed(st.iterations()); \
//     }                                                                       \
//     BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMillisecond)->Iterations(2);    
    

// #define BM_BODY_OP_TIME(OP)                                                 \
//     BENCHMARK_DEFINE_F(SCMPolarisFixture, OP)(benchmark::State & st) {      \
//         for (auto _ : st) {                                                 \
//                 m_driver->OP(801*(1e-3));                                         \
//         }                                                                   \
//         st.SetItemsProcessed(st.iterations()); \
//     }                                                                       \
//     BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMicrosecond);      

    // #define BM_BODY_OP_TIME(OP)                                                 \
    // BENCHMARK_DEFINE_F(SCMPolarisFixture, OP){          \
    //            std::cout<<"sss"<<std::endl;                                 \
    //             m_system->OP(1e-3);                                         \
    //     st.SetItemsProcessed(st.iterations() * m_system->Get_bodylist().size()); \
    // }                                                                       \
    // BENCHMARK_REGISTER_F(SCMPolarisFixture, OP)->Unit(benchmark::kMicrosecond);   

// #define BM_BODY_OP_VOID(OP)                                                 \
//     BENCHMARK_DEFINE_F(SystemFixture, OP)(benchmark::State & st) {          \
//         for (auto _ : st) {                                                 \
//             for (auto body : sys->Get_bodylist()) {                         \
//                 body->OP();                                                 \
//             }                                                               \
//         }                                                                   \
//         st.SetItemsProcessed(st.iterations() * sys->Get_bodylist().size()); \
//     }                                                                       \
//     BENCHMARK_REGISTER_F(SystemFixture, OP)->Unit(benchmark::kMicrosecond);

// #define BM_BODY_OP_STEP(OP)                                                 \
//     BENCHMARK_DEFINE_F(SystemFixture, OP)(benchmark::State & st) {          \
//         for (auto _ : st) {                                                 \
//             for (auto body : sys->Get_bodylist()) {                         \
//                 body->OP(time_step);                                        \
//             }                                                               \
//         }                                                                   \
//         st.SetItemsProcessed(st.iterations() * sys->Get_bodylist().size()); \
//     }                                                                       \
//     BENCHMARK_REGISTER_F(SystemFixture, OP)->Unit(benchmark::kMicrosecond);

// // Benchmark individual operations

BM_DRIVER_OP_VOID(DriverGetInput)
BM_DRIVER_SYNCRONIZE_TIME(DriverSynchronize)
BM_TERRAIN_SYNCRONIZE_TIME(TerrainSynchronize)  
BM_VEHICLE_SYNCRONIZE_TIME(VehicleSynchronize)  
BM_SYSTEM_OP_TIMESTEP(DoStepDynamics)
BM_VIS_OP_TIMESTEP(VisAdvance)


// Benchmark all operations in a single loop
BENCHMARK_DEFINE_F(SCMPolarisFixture, SingleLoop1)(benchmark::State& st) {
    for (auto _ : st) {
        m_driver->Synchronize(current_time);
        
    }
    st.SetItemsProcessed(st.iterations());
}
BENCHMARK_REGISTER_F(SCMPolarisFixture, SingleLoop1)->Unit(benchmark::kMillisecond)->Iterations(1);  

// Benchmark all operations in a single loop
BENCHMARK_DEFINE_F(SCMPolarisFixture, SingleLoop2)(benchmark::State& st) {
    for (auto _ : st) {
        m_driver->Synchronize(current_time);
        m_terrain->Synchronize(current_time);
        m_vehicle->Synchronize(current_time, m_driver_inputs, *m_terrain);
        m_vis->Synchronize("", m_driver_inputs);
        
    }
    st.SetItemsProcessed(st.iterations());
}
BENCHMARK_REGISTER_F(SCMPolarisFixture, SingleLoop2)->Unit(benchmark::kMillisecond)->Iterations(1);

// Benchmark all operations in a single loop
BENCHMARK_DEFINE_F(SCMPolarisFixture, SingleLoop3)(benchmark::State& st) {
    for (auto _ : st) {
        m_driver->Synchronize(current_time);
        m_terrain->Synchronize(current_time);
        m_vehicle->Synchronize(current_time, m_driver_inputs, *m_terrain);
        m_vis->Synchronize("", m_driver_inputs);
        // Advance dynamics
        m_system->DoStepDynamics(time_step);
        
    }
    st.SetItemsProcessed(st.iterations());
}
BENCHMARK_REGISTER_F(SCMPolarisFixture, SingleLoop3)->Unit(benchmark::kMillisecond)->Iterations(1);

// Benchmark all operations in a single loop
BENCHMARK_DEFINE_F(SCMPolarisFixture, SingleLoop4)(benchmark::State& st) {
    for (auto _ : st) {
        m_driver->Synchronize(current_time);
        m_terrain->Synchronize(current_time);
        m_vehicle->Synchronize(current_time, m_driver_inputs, *m_terrain);
        m_vis->Synchronize("", m_driver_inputs);
        // Advance dynamics
        m_system->DoStepDynamics(time_step);
        m_vis->Advance(step_size);
    }
    st.SetItemsProcessed(st.iterations());
}
BENCHMARK_REGISTER_F(SCMPolarisFixture, SingleLoop4)->Unit(benchmark::kMillisecond)->Iterations(1);


// BM_DRIVER_OP_TIME(Synchronize)
// BM_BODY_OP_TIME(Synchronize)
// BM_BODY_OP_TIME(UpdateForces)
// BM_BODY_OP_TIME(UpdateMarkers)
// BM_BODY_OP_VOID(ClampSpeed)
// BM_BODY_OP_VOID(ComputeGyro)

// Benchmark all operations in a single loop
// BENCHMARK_DEFINE_F(SystemFixture, SingleLoop)(benchmark::State& st) {
//     for (auto _ : st) {
//         for (auto body : sys->Get_bodylist()) {
//             std::cout<<"Here in the body"<<std::endl;
//             body->UpdateTime(current_time);
//             body->UpdateForces(current_time);
//             body->UpdateMarkers(current_time);
//             body->ClampSpeed();
//             body->ComputeGyro();
//         }
//     }
//     st.SetItemsProcessed(st.iterations() * sys->Get_bodylist().size());
// }
// BENCHMARK_REGISTER_F(SystemFixture, SingleLoop)->Unit(benchmark::kMicrosecond);
////BENCHMARK_REGISTER_F(SystemFixture, SingleLoop)->Unit(benchmark::kMicrosecond)->Iterations(1);

////BENCHMARK_MAIN();
