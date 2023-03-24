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


using namespace chrono;
using namespace chrono::vehicle;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------

class CustomTerrain : public ChTerrain {
  public:
    CustomTerrain(ChSystem& sys, WheeledVehicle& vehicle);
    void Create(const std::string& terrain_dir, bool vis = true);
    void Synchronize(double time, int frame, bool debug_output);
    virtual void Advance(double step) override;

  private:
    ChSystem& m_sys;
    const WheeledVehicle& m_vehicle;
    std::array<std::shared_ptr<ChWheel>, 4> m_wheels;
    std::shared_ptr<ChParticleCloud> m_particles;
    
    double m_mbs_inputs[1000][90];

    std::array<TerrainForce, 4> m_tire_forces;

};

// -----------------------------------------------------------------------------

CustomTerrain::CustomTerrain(ChSystem& sys, WheeledVehicle& vehicle) : m_sys(sys), m_vehicle(vehicle)  {
    m_wheels[0] = vehicle.GetWheel(0, LEFT);
    m_wheels[1] = vehicle.GetWheel(0, RIGHT);
    m_wheels[2] = vehicle.GetWheel(1, LEFT);
    m_wheels[3] = vehicle.GetWheel(1, RIGHT);
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


    int row=0;
    std::string line2;
    std::string cell2;
    std::ifstream is2(vehicle::GetDataFile(terrain_dir + "/mbs.txt"));
    while (getline(is2, line2)) {
        std::stringstream ls2(line2);
        for (int i = 0; i < 90; i++) {
            getline(ls2, cell2, ',');
            m_mbs_inputs[row][i]  = stod(cell2);
            
        }
        row++;
    }
    is2.close();

    m_sys.Add(m_particles);

    if (vis) {
        auto sph = chrono_types::make_shared<ChSphereShape>();
        sph->GetSphereGeometry().rad = 0.01;
        m_particles->AddVisualShape(sph);
    }

}



void CustomTerrain::Synchronize(double time,int frame, bool debug_output) {

    for (int i = 0; i < 4; i++) {

        ChVector<> disc_center = m_wheels[i]->GetPos();

        // Call the forces and torques
        double Fx = m_mbs_inputs[frame][66+6*i];
        double Fy = m_mbs_inputs[frame][66+6*i+1];
        double Fz = m_mbs_inputs[frame][66+6*i+2];
        double Mx = m_mbs_inputs[frame][66+6*i+3];
        double My = m_mbs_inputs[frame][66+6*i+4];
        double Mz = m_mbs_inputs[frame][66+6*i+5];

        if (debug_output)
         std::cout<<"time= "<<time<<"frame= "<<frame<<"i= "<<i<<"Fx= "<<Fx<<"Fy= "<<Fy<<"Fz= "<<Fz<<"Mx= "<<Mx<<"My= "<<My<<"Mz= "<<Mz<<std::endl;
   
        // Tire force and moment in tire frame
        ChVector<> tire_F(Fx, Fy, Fz);
        ChVector<> tire_M(Mx, My, Mz);

        // Load the tire force structure (all expressed in absolute frame)
        m_tire_forces[i].force = tire_F;
        m_tire_forces[i].moment = tire_M;
        m_tire_forces[i].point = disc_center;
    }

    for (int i = 0; i < 4; i++) {
        m_wheels[i]->GetSpindle()->Accumulate_force(m_tire_forces[i].force, m_tire_forces[i].point, false);
        m_wheels[i]->GetSpindle()->Accumulate_torque(m_tire_forces[i].moment, false);
    }

}

void CustomTerrain::Advance(double step) {}

// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse arguments
    std::string terrain_dir = "terrain/scm/testterrain";
    double tend = 1.0;
    bool run_time_vis = true;
    bool debug_output = false;

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
    terrain.Create(terrain_dir,true);

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

        DriverInputs driver_inputs = {0.0, 0.0, 0.0};

        // Synchronize subsystems
        vehicle.Synchronize(t, driver_inputs, terrain);
        terrain.Synchronize(t,frame,debug_output);

        // Advance system state
        terrain.Advance(step_size);
        vis.Advance(step_size);
        sys.DoStepDynamics(step_size);
        t += step_size;

        frame++;
    }

    return 0;
}