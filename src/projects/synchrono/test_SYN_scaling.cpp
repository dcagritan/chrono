// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2020 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Jay Taves
// =============================================================================
//
// Demo code illustrating synchronization of the SCM semi-empirical model for
// deformable soil
//
// See also in chrono_vehicle:
// - demo_VEH_DeformableSoil
// - demo_VEH_DeformableSoilAndTire
// - demo_VEH_HMMWV_DefSoil
//
// =============================================================================

#include "chrono/core/ChRealtimeStep.h"

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/SCMTerrain.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/utils/ChVehiclePath.h"

#include "chrono_models/vehicle/hmmwv/HMMWV.h"

#include "chrono_synchrono/SynConfig.h"
#include "chrono_synchrono/SynChronoManager.h"
#include "chrono_synchrono/agent/SynWheeledVehicleAgent.h"
#include "chrono_synchrono/agent/SynSCMTerrainAgent.h"
#include "chrono_synchrono/communication/mpi/SynMPICommunicator.h"
#include "chrono_synchrono/utils/SynDataLoader.h"
#include "chrono_synchrono/utils/SynLog.h"

#ifdef CHRONO_IRRLICHT
    #include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"
#endif

#include "chrono_thirdparty/cxxopts/ChCLI.h"

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::synchrono;
using namespace chrono::vehicle::hmmwv;

using std::cout;
using std::endl;
using std::cin;

// =============================================================================

double terrainLength = 50;  // size in X direction
double terrainWidth = 50;   // size in Y direction
double delta = 0.05;        // SCM grid spacing

double target_speed = 10;

// Simulation run time
double end_time = 10;

// Simulation step size
double step_size = 2e-3;

// Number of threads
int nthreads = 4;

// Moving patches under each wheel
bool wheel_patches = false;

// Better conserve mass by displacing soil to the sides of a rut
const bool bulldozing = false;

// How often SynChrono state messages are interchanged
double heartbeat = 1e-2;  // 100[Hz]

// Rank for run-time visualization
int vis_rank = -1;

// =============================================================================

// Forward declares for straight forward helper functions
void AddCommandLineOptions(ChCLI& cli);
void PrintStepStatistics(std::ostream& os, ChSystem* sys);

// =============================================================================

int main(int argc, char* argv[]) {
    // -----------------------
    // Create SynChronoManager
    // -----------------------
    auto communicator = chrono_types::make_shared<SynMPICommunicator>(argc, argv);
    int node_id = communicator->GetRank();
    int num_nodes = communicator->GetNumRanks();
    SynChronoManager syn_manager(node_id, num_nodes, communicator);

    // Copyright
    if (node_id == 0) {
        SynLog() << "Copyright (c) 2020 projectchrono.org\n";
        SynLog() << "Chrono version: " << CHRONO_VERSION << "\n\n";
    }

#ifdef _DEBUG
    if (node_id == 0) {
        int foo;
        cout << "Enter something to continue..." << endl;
        cin >> foo;
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // ------------------------------------------------
    // CLI SETUP - Get parameters from the command line
    // ------------------------------------------------

    ChCLI cli(argv[0]);

    AddCommandLineOptions(cli);
    if (!cli.Parse(argc, argv, node_id == 0))
        return 0;

    // Normal simulation options
    step_size = cli.GetAsType<double>("step_size");
    end_time = cli.GetAsType<double>("end_time");
    heartbeat = cli.GetAsType<double>("heartbeat");
    nthreads = cli.GetAsType<int>("nthreads");
    wheel_patches = cli.GetAsType<bool>("wheel_patches");

    vis_rank = cli.GetAsType<int>("vis");
#if !defined(CHRONO_IRRLICHT)
    if (vis_rank >= 0 && node_id == 0)
        cout << "Chrono::Irrlicht not available. Disabling visualization." << endl;
    vis_rank = -1;
#endif
    bool visualize = (vis_rank == node_id);

    // Change SynChronoManager settings
    syn_manager.SetHeartbeat(heartbeat);

    // ----------------------
    // Vehicle Initialization
    // ----------------------
    // Calculate initial position and paths for each vehicle
    double pathLength = 1.5 * target_speed * end_time;

    ChVector<> init_loc;
    ChQuaternion<> init_rot;
    std::shared_ptr<ChBezierCurve> path;
    if (node_id % 2 == 0) {
        // Start even vehicles in a row on the south side, driving north
        init_loc = ChVector<>(0, 2.0 * (node_id + 1), 0.5);
        init_rot = Q_from_AngZ(0);
        path = StraightLinePath(init_loc, init_loc + ChVector<>(pathLength, 0, 0));
    } else {
        // Start odd vehicles staggered going up the west edge, driving east
        init_loc = ChVector<>(2.0 * (node_id - 1), -5.0 - 2.0 * (node_id - 1), 0.5);
        init_rot = Q_from_AngZ(CH_C_PI / 2);
        path = StraightLinePath(init_loc, init_loc + ChVector<>(0, pathLength, 0));
    }

    // Create the HMMWV
    HMMWV_Full hmmwv;
    hmmwv.SetContactMethod(ChContactMethod::SMC);
    hmmwv.SetChassisFixed(false);
    hmmwv.SetInitPosition(ChCoordsys<>(init_loc, init_rot));
    hmmwv.SetPowertrainType(PowertrainModelType::SHAFTS);
    hmmwv.SetDriveType(DrivelineTypeWV::AWD);
    hmmwv.SetTireType(TireModelType::RIGID);
    hmmwv.SetTireStepSize(step_size);
    hmmwv.Initialize();

    hmmwv.SetChassisVisualizationType(VisualizationType::NONE);
    hmmwv.SetSuspensionVisualizationType(VisualizationType::MESH);
    hmmwv.SetSteeringVisualizationType(VisualizationType::NONE);
    hmmwv.SetWheelVisualizationType(VisualizationType::MESH);
    hmmwv.SetTireVisualizationType(VisualizationType::MESH);

    // -----------------------------------------------------------
    // Set tire contact material, contact model, and visualization
    // -----------------------------------------------------------
    auto wheel_material = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    wheel_material->SetFriction(0.8f);
    wheel_material->SetYoungModulus(1.0e6f);
    wheel_material->SetRestitution(0.1f);

    hmmwv.SetTireVisualizationType(VisualizationType::MESH);

    // What we defined earlier, a straight line
    ChPathFollowerDriver driver(hmmwv.GetVehicle(), path, "Box path", target_speed);
    driver.Initialize();

    // Reasonable defaults for the underlying PID
    driver.GetSpeedController().SetGains(0.4, 0, 0);
    driver.GetSteeringController().SetGains(0.4, 0.1, 0.2);
    driver.GetSteeringController().SetLookAheadDistance(2);

    // Add vehicle as an agent
    auto vehicle_agent = chrono_types::make_shared<SynWheeledVehicleAgent>(&hmmwv.GetVehicle());
    vehicle_agent->SetZombieVisualizationFiles("hmmwv/hmmwv_chassis.obj", "hmmwv/hmmwv_rim.obj",
                                               "hmmwv/hmmwv_tire_left.obj");
    vehicle_agent->SetNumWheels(4);
    syn_manager.AddAgent(vehicle_agent);

    // ----------------------
    // Terrain specific setup
    // ----------------------
    ChSystem* system = hmmwv.GetSystem();
    system->SetNumThreads(nthreads, 1, 1);
    if (node_id == 0)
        std::cout << "Num threads: " << nthreads << std::endl;

    SCMTerrain terrain(system, visualize);
    terrain.SetSoilParameters(2e6,   // Bekker Kphi
                              0,     // Bekker Kc
                              1.1,   // Bekker n exponent
                              0,     // Mohr cohesive limit (Pa)
                              30,    // Mohr friction limit (degrees)
                              0.01,  // Janosi shear coefficient (m)
                              2e8,   // Elastic stiffness (Pa/m), before plastic yield
                              3e4    // Damping (Pa s/m), proportional to negative vertical speed (optional)
    );

    // Configure the SCM terrain
    if (bulldozing) {
        terrain.EnableBulldozing(bulldozing);
        terrain.SetBulldozingParameters(
            55,   // angle of friction for erosion of displaced material at the border of the rut
            0.8,  // displaced material vs downward pressed material.
            5,    // number of erosion refinements per timestep
            10);  // number of concentric vertex selections subject to erosion
    }

    if (wheel_patches) {
        // Optionally, enable moving patch feature (multiple patches around each wheel)
        for (auto& axle : hmmwv.GetVehicle().GetAxles()) {
            terrain.AddMovingPatch(axle->m_wheels[0]->GetSpindle(), ChVector<>(0, 0, 0), ChVector<>(1, 0.5, 1));
            terrain.AddMovingPatch(axle->m_wheels[1]->GetSpindle(), ChVector<>(0, 0, 0), ChVector<>(1, 0.5, 1));
        }
    } else {
        // Optionally, enable moving patch feature (single patch around vehicle chassis)
        terrain.AddMovingPatch(hmmwv.GetChassisBody(), ChVector<>(0, 0, 0), ChVector<>(5, 3, 1));
    }

    terrain.SetPlotType(vehicle::SCMTerrain::PLOT_SINKAGE, 0, 0.1);

    terrain.Initialize(terrainLength, terrainWidth, delta);

    // Create an SCMTerrainAgent and add it to the SynChrono manager
    auto scm = chrono_types::make_shared<SCMTerrain>(terrain);
    auto terrain_agent = chrono_types::make_shared<SynSCMTerrainAgent>(scm);
    syn_manager.AddAgent(terrain_agent);

    // Initialzie the SynChrono manager
    syn_manager.Initialize(system);

    // -------------
    // Visualization
    // -------------
#ifdef CHRONO_IRRLICHT
    // Create the vehicle Irrlicht interface
    std::shared_ptr<ChWheeledVehicleVisualSystemIrrlicht> vis;
    if (visualize) {
        vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
        vis->SetWindowTitle("SynChrono SCM test");
        vis->SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), 6.0, 0.5);
        vis->Initialize();
        vis->AddTypicalLights();
        vis->AddSkyBox();
        vis->AddLogo();
        vis->AttachVehicle(&hmmwv.GetVehicle());
    }

    double render_step_size = 1.0 / 100;
    int render_steps = (int)std::ceil(render_step_size / step_size);
#endif

    // ---------------
    // Simulation loop
    // ---------------
    bool stats_done = false;

    system->SetSolverMaxIterations(50);

    // Initialize simulation frame counters
    int step_number = 0;
    ChRealtimeStepTimer realtime_timer;

    double chrono_step = 0;

    ChTimer timer;
    timer.start();

    while (true) {
        double time = system->GetChTime();

        if (!syn_manager.IsOk()) {
            if (node_id == 0)
                cout << "SynChronoManager has shutdown!" << endl;
            break;
        }

        if (time > end_time) {
            if (!stats_done) {
                timer.stop();
                for (int i = 0; i < num_nodes; i++) {
                    if (node_id == i) {
                        cout << endl;
                        cout << "stop timer at: " << end_time << endl;
                        cout << "elapsed time:  " << timer() << endl;
                        cout << "chrono solver: " << chrono_step << endl;
                        cout << "RTF:           " << (timer() / end_time) << endl;
                        cout << "\nSCM stats for last step:" << endl;
                        terrain.PrintStepStatistics(cout);
                        cout << "\nChrono stats for last step:" << endl;
                        PrintStepStatistics(cout, system);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                stats_done = true;
            }

            if (vis_rank == -1)
                break;
        }

#ifdef CHRONO_IRRLICHT
        if (vis && !vis->Run())  //  Irrlicht visualization has stopped
            break;

        // Render scene
        if (vis && step_number % render_steps == 0) {
            vis->BeginScene();
            vis->Render();
            vis->EndScene();
        }
#endif

        // Get driver inputs
        DriverInputs driver_inputs = driver.GetInputs();

        // Synchronize between nodes
        syn_manager.Synchronize(time);

        // Update modules (process inputs from other modules)
        driver.Synchronize(time);
        terrain.Synchronize(time);
        hmmwv.Synchronize(time, driver_inputs, terrain);
#ifdef CHRONO_IRRLICHT
        if (vis)
            vis->Synchronize(time, driver_inputs);
#endif

        driver.Advance(step_size);
        terrain.Advance(step_size);
        hmmwv.Advance(step_size);
#ifdef CHRONO_IRRLICHT
        if (vis)
            vis->Advance(step_size);
#endif

        chrono_step += system->GetTimerStep();

        // Increment frame number
        step_number++;

        // Spin in place for real time to catch up
        ////realtime_timer.Spin(step_size);
    }

    return 0;
}

void AddCommandLineOptions(ChCLI& cli) {
    cli.AddOption<double>("Test", "s,step_size", "Step size", std::to_string(step_size));
    cli.AddOption<double>("Test", "e,end_time", "End time", std::to_string(end_time));
    cli.AddOption<double>("Test", "b,heartbeat", "Heartbeat", std::to_string(heartbeat));
    cli.AddOption<int>("Test", "n,nthreads", "Number threads", std::to_string(nthreads));
    cli.AddOption<bool>("Test", "w,wheel_patches", "Use patches under each wheel", "false");
    cli.AddOption<int>("Test", "v,vis", "Run-time visualization rank", std::to_string(vis_rank));
}

void PrintStepStatistics(std::ostream& os, ChSystem* sys) {
    os << " Step (ms):    " << 1e3 * sys->GetTimerStep() << std::endl;
    os << "   Advance:    " << 1e3 * sys->GetTimerAdvance() << std::endl;
    os << "   LSsolve:    " << 1e3 * sys->GetTimerLSsolve() << std::endl;
    os << "   LSsetup:    " << 1e3 * sys->GetTimerLSsetup() << std::endl;
    os << "   Jacobian:   " << 1e3 * sys->GetTimerJacobian() << std::endl;
    os << "   Collision:  " << 1e3 * sys->GetTimerCollision() << std::endl;
    os << "   Setup:      " << 1e3 * sys->GetTimerSetup() << std::endl;
    os << "   Update:     " << 1e3 * sys->GetTimerUpdate() << std::endl;
}
