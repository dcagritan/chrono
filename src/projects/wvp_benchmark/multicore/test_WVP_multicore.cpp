// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Radu Serban
// =============================================================================
//
// Sample test program for multicore WVP simulation.
//
// Contact uses non-smooth (DVI) formulation.
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
// All units SI.
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>

#include "chrono/ChConfig.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/ChDriver.h"
#include "chrono_vehicle/terrain/FlatTerrain.h"

#include "chrono_multicore/physics/ChSystemMulticore.h"

#include "chrono_opengl/ChVisualSystemOpenGL.h"

#include "chrono_models/vehicle/wvp/WVP.h"

using namespace chrono;
using namespace chrono::collision;
using namespace chrono::vehicle;
using namespace chrono::vehicle::wvp;

// -----------------------------------------------------------------------------
// Specification of the terrain
// -----------------------------------------------------------------------------

// Container
double hdimX = 6;
double hdimY = 1.75;
double hdimZ = 0.5;
double hthick = 0.05;

// Number of layers of granular material
int num_layers = 4;

// -----------------------------------------------------------------------------
// Specification of the vehicle model
// -----------------------------------------------------------------------------

// Initial vehicle position, orientation, and forward velocity
ChVector<> initLoc(-hdimX + 2.5, 0, 0.6);
ChQuaternion<> initRot(1, 0, 0, 0);
double initSpeed = 0;

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Total simulation duration.
double time_end = 7;

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 20;
bool thread_tuning = false;

// Integration step size
double time_step = 1e-3;

// Solver settings
int max_iteration_bilateral = 100;
int max_iteration_normal = 0;
int max_iteration_sliding = 50;
int max_iteration_spinning = 0;

double tolerance = 1e-3;
float contact_recovery_speed = 1000;

// =============================================================================

class My_Driver : public ChDriver {
  public:
    My_Driver(ChVehicle& vehicle) : ChDriver(vehicle) {}

    virtual void Synchronize(double time) override {
        m_braking = 0;
        m_throttle = 0.5;
        m_steering = 0;
    }
};

// =============================================================================

void CreateContainer(ChSystem* system, float mu_g, float coh_g) {
    bool visible_walls = false;

    auto material = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    material->SetFriction(mu_g);
    material->SetCompliance(1e-9f);
    material->SetCohesion(coh_g);

    auto ground = chrono_types::make_shared<ChBody>(ChCollisionSystemType::CHRONO);
    ground->SetIdentifier(-1);
    ground->SetMass(1000);
    ground->SetBodyFixed(true);
    ground->SetCollide(true);

    ground->GetCollisionModel()->ClearModel();

    // Attention: collision family for ground should be >= 5 (first values used in Chrono::Vehicle)
    ground->GetCollisionModel()->SetFamily(5);
    ground->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(WheeledCollisionFamily::TIRES);

    // Bottom box
    utils::AddBoxGeometry(ground.get(), material, ChVector<>(hdimX, hdimY, hthick), ChVector<>(0, 0, -hthick),
                          ChQuaternion<>(1, 0, 0, 0), true);
    // Left box
    utils::AddBoxGeometry(ground.get(), material, ChVector<>(hdimX, hthick, hdimZ + hthick),
                          ChVector<>(0, hdimY + hthick, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);
    // Right box
    utils::AddBoxGeometry(ground.get(), material, ChVector<>(hdimX, hthick, hdimZ + hthick),
                          ChVector<>(0, -hdimY - hthick, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);

    // Front box
    utils::AddBoxGeometry(ground.get(), material, ChVector<>(hthick, hdimY, hdimZ + hthick),
                          ChVector<>(hdimX + hthick, 0, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);
    // Rear box
    utils::AddBoxGeometry(ground.get(), material, ChVector<>(hthick, hdimY, hdimZ + hthick),
                          ChVector<>(-hdimX - hthick, 0, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);

    ground->GetCollisionModel()->BuildModel();

    system->AddBody(ground);
}

// =============================================================================

int CreateParticles(ChSystem* system, double r_g, double rho_g, float mu_g, float coh_g) {
    // Create a material
    auto mat_g = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    mat_g->SetFriction(mu_g);
    mat_g->SetRestitution(0);
    mat_g->SetCohesion(coh_g);

    // Create a particle generator and a mixture entirely made out of spheres
    double r = 1.01 * r_g;
    utils::PDSampler<double> sampler(2 * r);
    utils::Generator gen(system);
    std::shared_ptr<utils::MixtureIngredient> m1 = gen.AddMixtureIngredient(utils::MixtureType::SPHERE, 1.0);
    m1->setDefaultMaterial(mat_g);
    m1->setDefaultDensity(rho_g);
    m1->setDefaultSize(r_g);

    // Set starting value for body identifiers
    gen.setBodyIdentifier(1);

    // Create particles in layers until reaching the desired number of particles
    ChVector<> hdims(hdimX - r, hdimY - r, 0);
    ChVector<> center(0, 0, 2 * r);

    for (int il = 0; il < num_layers; il++) {
        gen.CreateObjectsBox(sampler, center, hdims);
        center.z() += 2 * r;
    }

    return gen.getTotalNumBodies();
}

double FindHighestParticle(ChSystem* system) {
    double highest = 0;
    for (auto body : system->Get_bodylist()) {
        if (body->GetIdentifier() > 0 && body->GetPos().z() > highest)
            highest = body->GetPos().z();
    }
    return highest;
}

// =============================================================================

int main(int argc, char* argv[]) {
    double slope = 0;
    double r_g = 0.03;
    double rho_g = 2000;
    float coh_g = 100;
    float mu_g = 0.8f;

    bool render = true;

    // -------------
    // Create system
    // -------------

    ChSystemMulticoreNSC* system = new ChSystemMulticoreNSC();
    system->Set_G_acc(ChVector<>(0, 0, -9.81));

    // ---------------------
    // Set number of threads
    // ---------------------

    int max_threads = omp_get_num_procs();
    if (threads > max_threads)
        threads = max_threads;
    if (thread_tuning)
        system->SetNumThreads(1, 1, threads);
    else
        system->SetNumThreads(threads);

    // --------------------
    // Edit system settings
    // --------------------

    system->GetSettings()->solver.tolerance = tolerance;
    system->GetSettings()->solver.solver_mode = SolverMode::SLIDING;
    system->GetSettings()->solver.max_iteration_normal = max_iteration_normal;
    system->GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;
    system->GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;
    system->GetSettings()->solver.max_iteration_bilateral = max_iteration_bilateral;
    system->GetSettings()->solver.compute_N = false;
    system->GetSettings()->solver.alpha = 0;
    system->GetSettings()->solver.cache_step_length = true;
    system->GetSettings()->solver.use_full_inertia_tensor = false;
    system->GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;
    system->GetSettings()->solver.bilateral_clamp_speed = 1e8;
    system->ChangeSolverType(SolverType::BB);

    system->GetSettings()->collision.collision_envelope = 0.1 * r_g;
    system->GetSettings()->collision.narrowphase_algorithm = ChNarrowphase::Algorithm::HYBRID;
    system->GetSettings()->collision.bins_per_axis = vec3(100, 30, 2);

    // Specify active box.
    system->GetSettings()->collision.use_aabb_active = false;
    system->GetSettings()->collision.aabb_min = real3(-1.1 * hdimX, -1.1 * hdimY, 0);
    system->GetSettings()->collision.aabb_max = real3(+1.1 * hdimX, +1.1 * hdimY, 10 * hdimZ);

    // ------------------
    // Create the terrain
    // ------------------

    CreateContainer(system, mu_g, coh_g);
    int actual_num_particles = CreateParticles(system, r_g, rho_g, mu_g, coh_g);
    std::cout << "Created " << actual_num_particles << " particles." << std::endl;

    FlatTerrain terrain(0);

    // ------------------
    // Create the vehicle
    // ------------------

    WVP wvp(system);

    wvp.SetContactMethod(ChContactMethod::NSC);
    wvp.SetChassisFixed(false);
    wvp.SetInitPosition(ChCoordsys<>(initLoc + ChVector<>(0, 0, 0.3), initRot));
    wvp.SetInitFwdVel(initSpeed);
    wvp.SetTireType(TireModelType::RIGID);

    wvp.Initialize();

    wvp.SetChassisVisualizationType(VisualizationType::NONE);
    wvp.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    wvp.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    wvp.SetWheelVisualizationType(VisualizationType::PRIMITIVES);
    wvp.SetTireVisualizationType(VisualizationType::PRIMITIVES);

    // -----------------
    // Create the driver
    // -----------------

    My_Driver driver(wvp.GetVehicle());

    // -----------------
    // Initialize OpenGL
    // -----------------

    opengl::ChVisualSystemOpenGL vis;
    if (render) {
        vis.AttachSystem(system);
        vis.SetWindowTitle("Test");
        vis.SetWindowSize(1280, 720);
        vis.SetRenderMode(opengl::WIREFRAME);
        vis.Initialize();
        vis.AddCamera(ChVector<>(0, -10, 0), ChVector<>(0, 0, 0));
        vis.SetCameraVertical(CameraVerticalDir::Z);
    }

    // ---------------
    // Simulation loop
    // ---------------

    double time = 0;
    double exec_time = 0;

    while (time < time_end) {
        // Driver inputs
        DriverInputs driver_inputs = driver.GetInputs();

        // Synchronize vehicle systems
        driver.Synchronize(time);
        wvp.Synchronize(time, driver_inputs, terrain);

        // Advance vehicle systems
        driver.Advance(time_step);
        wvp.Advance(time_step);

        if (render) {
            if (vis.Run()) {
                vis.Render();
            } else {
                break;
            }
        }

        // Update counters.
        time += time_step;
        exec_time += system->GetTimerStep();
    }

    // Final stats
    std::cout << "==================================" << std::endl;
    std::cout << "Simulation time:   " << exec_time << std::endl;
    std::cout << "Number of threads: " << threads << std::endl;

    return 0;
}
