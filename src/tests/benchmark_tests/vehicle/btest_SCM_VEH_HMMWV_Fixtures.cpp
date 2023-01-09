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
// Authors: Radu Serban, Deniz Cagri Tanyildiz
// =============================================================================
//
// Benchmark test for HMMWV on SCM terrain.
//
// =============================================================================

#include "chrono/utils/ChBenchmark.h"
#include "chrono/physics/ChBodyEasy.h"
#include <benchmark/benchmark.h>

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"
#include "chrono_vehicle/utils/ChVehiclePath.h"

#include "chrono_models/vehicle/hmmwv/HMMWV.h"

#ifdef CHRONO_IRRLICHT
    #include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"
#endif

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::hmmwv;

// =============================================================================


double patch_size = 50.0;
int num_div = 1000;

// =============================================================================

class HmmwvScmDriver : public ChDriver {
  public:
    HmmwvScmDriver(ChVehicle& vehicle, double delay) : ChDriver(vehicle), m_delay(delay) {}
    ~HmmwvScmDriver() {}

    virtual void Synchronize(double time) override {
        m_throttle = 0;
        m_steering = 0;
        m_braking = 0;

        double eff_time = time - m_delay;

        // Do not generate any driver inputs for a duration equal to m_delay.
        if (eff_time < 0)
            return;

        if (eff_time > 0.2)
            m_throttle = 0.7;
        else
            m_throttle = 3.5 * eff_time;

        if (eff_time < 2)
            m_steering = 0;
        else
            m_steering = 0.6 * std::sin(CH_C_2PI * (eff_time - 2) / 6);
    }

  private:
    double m_delay;
};

// =============================================================================

class HmmwvScmFixtureTest : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& st) override;
    void TearDown(const ::benchmark::State&) override;

    ChSystem* GetSystem() { return m_hmmwv->GetSystem(); }
    void ExecuteStep();

    void SimulateVis();

    double GetTime() const { return m_hmmwv->GetSystem()->GetChTime(); }
    double GetLocation() const { return m_hmmwv->GetVehicle().GetPos().x(); }

  private:
    HMMWV_Full* m_hmmwv;
    HmmwvScmDriver* m_driver;
    SCMDeformableTerrain* m_terrain;

    double m_step;
};

void HmmwvScmFixtureTest::SetUp(const ::benchmark::State& st) {
    m_step=2e-3;
    PowertrainModelType powertrain_model = PowertrainModelType::SHAFTS;
    DrivelineTypeWV drive_type = DrivelineTypeWV::AWD;
    TireModelType tire_type = TireModelType::RIGID_MESH;
    VisualizationType tire_vis = VisualizationType::MESH;

    // Create the HMMWV vehicle, set parameters, and initialize.
    m_hmmwv = new HMMWV_Full();
    m_hmmwv->SetContactMethod(ChContactMethod::SMC);
    m_hmmwv->SetChassisFixed(false);
    m_hmmwv->SetInitPosition(
        ChCoordsys<>(ChVector<>(5.0 - patch_size / 2, 5.0 - patch_size / 2, 0.7), Q_from_AngZ(CH_C_PI / 4)));
    m_hmmwv->SetPowertrainType(powertrain_model);
    m_hmmwv->SetDriveType(drive_type);
    m_hmmwv->SetTireType(tire_type);
    m_hmmwv->SetTireStepSize(m_step);
    m_hmmwv->SetAerodynamicDrag(0.5, 5.0, 1.2);
    m_hmmwv->Initialize();

    m_hmmwv->SetChassisVisualizationType(VisualizationType::PRIMITIVES);
    m_hmmwv->SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    m_hmmwv->SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    m_hmmwv->SetWheelVisualizationType(VisualizationType::NONE);
    m_hmmwv->SetTireVisualizationType(tire_vis);

    m_hmmwv->GetSystem()->SetNumThreads(4);

    // Create the terrain using 4 moving patches
    m_terrain = new SCMDeformableTerrain(m_hmmwv->GetSystem());
    m_terrain->SetSoilParameters(2e6,   // Bekker Kphi
                                 0,     // Bekker Kc
                                 1.1,   // Bekker n exponent
                                 0,     // Mohr cohesive limit (Pa)
                                 30,    // Mohr friction limit (degrees)
                                 0.01,  // Janosi shear coefficient (m)
                                 2e8,   // Elastic stiffness (Pa/m), before plastic yield
                                 3e4    // Damping (Pa s/m), proportional to negative vertical speed (optional)
    );

    m_terrain->AddMovingPatch(m_hmmwv->GetVehicle().GetAxle(0)->GetWheel(VehicleSide::LEFT)->GetSpindle(),
                              ChVector<>(0, 0, 0), ChVector<>(1.0, 0.3, 1.0));
    m_terrain->AddMovingPatch(m_hmmwv->GetVehicle().GetAxle(0)->GetWheel(VehicleSide::RIGHT)->GetSpindle(),
                              ChVector<>(0, 0, 0), ChVector<>(1.0, 0.3, 1.0));
    m_terrain->AddMovingPatch(m_hmmwv->GetVehicle().GetAxle(1)->GetWheel(VehicleSide::LEFT)->GetSpindle(),
                              ChVector<>(0, 0, 0), ChVector<>(1.0, 0.3, 1.0));
    m_terrain->AddMovingPatch(m_hmmwv->GetVehicle().GetAxle(1)->GetWheel(VehicleSide::RIGHT)->GetSpindle(),
                              ChVector<>(0, 0, 0), ChVector<>(1.0, 0.3, 1.0));

    m_terrain->SetPlotType(vehicle::SCMDeformableTerrain::PLOT_SINKAGE, 0, 0.1);

    m_terrain->Initialize(patch_size, patch_size, patch_size / num_div);

    // Custom driver
    m_driver = new HmmwvScmDriver(m_hmmwv->GetVehicle(), 1.0);
    m_driver->Initialize();
}

void HmmwvScmFixtureTest::TearDown(const ::benchmark::State&) {
    delete m_hmmwv;
    delete m_terrain;
    delete m_driver;
}

void HmmwvScmFixtureTest::ExecuteStep() {
    double time = m_hmmwv->GetSystem()->GetChTime();
    std::cout<<"time= "<<time<<std::endl;

    // Driver inputs
    DriverInputs driver_inputs = m_driver->GetInputs();

    // Update modules (process inputs from other modules)
    m_driver->Synchronize(time);
    m_terrain->Synchronize(time);
    m_hmmwv->Synchronize(time, driver_inputs, *m_terrain);

    // Advance simulation for one timestep for all modules
    m_driver->Advance(m_step);
    m_terrain->Advance(m_step);
    m_hmmwv->Advance(m_step);
}

void HmmwvScmFixtureTest::SimulateVis() {
#ifdef CHRONO_IRRLICHT
    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->AttachVehicle(&m_hmmwv->GetVehicle());
    vis->SetWindowTitle("HMMWV SCM benchmark");
    vis->SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), 6.0, 0.5);
    vis->Initialize();
    vis->AddTypicalLights();

    while (vis->Run()) {
        DriverInputs driver_inputs = m_driver->GetInputs();

        vis->BeginScene();
        vis->Render();
        ExecuteStep();
        vis->Synchronize("SCM test", driver_inputs);
        vis->Advance(m_step);
        vis->EndScene();
    }
#endif
}

// =============================================================================

// Utility macros for benchmarking body operations with different signatures
#define BM_EXECUTION_TIME(OP)                                                 \
    BENCHMARK_DEFINE_F(HmmwvScmFixtureTest, OP)(benchmark::State & st) {        \
        for (auto _ : st) {                                                 \
            OP();                                            \
        }                                                                   \
        st.SetItemsProcessed(st.iterations());                              \
    }                                                                       \
    BENCHMARK_REGISTER_F(HmmwvScmFixtureTest, OP)->Unit(benchmark::kMillisecond);
BM_EXECUTION_TIME(ExecuteStep)
BM_EXECUTION_TIME(SimulateVis)