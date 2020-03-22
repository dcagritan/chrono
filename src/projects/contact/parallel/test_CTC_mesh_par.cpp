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
// Authors: Radu Serban
// =============================================================================
//
// Test mesh collision
//
// =============================================================================

#include "chrono/ChConfig.h"
#include "chrono/assets/ChColorAsset.h"
#include "chrono/assets/ChBoxShape.h"
#include "chrono/assets/ChSphereShape.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/solver/ChIterativeSolverParallel.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;

// ====================================================================================

int main(int argc, char* argv[]) {
    // ---------------------
    // Simulation parameters
    // ---------------------

    double gravity = 9.81;    // gravitational acceleration
    double time_step = 1e-4;  // integration step size

    double tolerance = 0;
    double contact_recovery_speed = 10;
    double collision_envelope = .005;

    uint max_iteration_normal = 0;
    uint max_iteration_sliding = 0;
    uint max_iteration_spinning = 100;
    uint max_iteration_bilateral = 0;
    
    enum class GeometryType { PRIMITIVE, MESH };
    GeometryType ground_geometry = GeometryType::PRIMITIVE;  // box or ramp mesh
    GeometryType object_geometry = GeometryType::MESH;       // sphere or tire mesh

    double mesh_swept_sphere_radius = 0.005;

    // ---------------------------
    // Contact material properties
    // ---------------------------

    ChMaterialSurface::ContactMethod contact_method = ChMaterialSurface::NSC;
    bool use_mat_properties = true;

    float object_friction = 0.9f;
    float object_restitution = 0.0f;
    float object_young_modulus = 2e7f;
    float object_poisson_ratio = 0.3f;
    float object_adhesion = 0.0f;
    float object_kn = 2e5;
    float object_gn = 40;
    float object_kt = 2e5;
    float object_gt = 20;

    float ground_friction = 0.9f;
    float ground_restitution = 0.01f;
    float ground_young_modulus = 1e6f;
    float ground_poisson_ratio = 0.3f;
    float ground_adhesion = 0.0f;
    float ground_kn = 2e5;
    float ground_gn = 40;
    float ground_kt = 2e5;
    float ground_gt = 20;

    // ---------------------------------
    // Parameters for the falling object
    // ---------------------------------

    ChVector<> pos(0.2, 0.55, 0.2);
    ChVector<> init_vel(0, 0, 0);
    ChVector<> init_omg(0, 0, 0);

    // ---------------------------------
    // Parameters for the containing bin
    // ---------------------------------

    double width = 2;
    double length = 1;
    double thickness = 0.1;

    // -----------------
    // Create the system
    // -----------------

    ChSystemParallel* system;

    switch (contact_method) {
        case ChMaterialSurface::NSC: {
            auto my_sys = new ChSystemParallelNSC();
            my_sys->ChangeSolverType(SolverType::APGD);
            my_sys->GetSettings()->solver.solver_mode = SolverMode::SPINNING;
            my_sys->GetSettings()->solver.max_iteration_normal = max_iteration_normal;
            my_sys->GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;
            my_sys->GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;
            my_sys->GetSettings()->solver.max_iteration_bilateral = max_iteration_bilateral;
            my_sys->GetSettings()->solver.alpha = 0;
            my_sys->GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;

            system = my_sys;
            break;
        }
        case ChMaterialSurface::SMC: {
            auto my_sys = new ChSystemParallelSMC();
            my_sys->GetSettings()->solver.contact_force_model = ChSystemSMC::Hertz;
            my_sys->GetSettings()->solver.tangential_displ_mode = ChSystemSMC::OneStep;

            system = my_sys;
            break;
        }
    }

    system->Set_G_acc(ChVector<>(0, -gravity, 0));

    int nthreads = 2;
    int max_threads = CHOMPfunctions::GetNumProcs();
    if (nthreads > max_threads)
        nthreads = max_threads;
    CHOMPfunctions::SetNumThreads(nthreads);

    system->GetSettings()->perform_thread_tuning = false;

    system->GetSettings()->solver.use_full_inertia_tensor = false;
    system->GetSettings()->solver.tolerance = tolerance;

    system->GetSettings()->collision.collision_envelope = collision_envelope;
    system->GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;
    system->GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    // Rotation Z->Y (because meshes used here assume Z up)
    ChQuaternion<> z2y = Q_from_AngX(-CH_C_PI_2);

    // Create the falling object
    auto object = std::shared_ptr<ChBody>(system->NewBody());
    system->AddBody(object);

    object->SetMass(200);
    object->SetInertiaXX(40.0 * ChVector<>(1, 1, 0.2));
    object->SetPos(pos);
    object->SetRot(z2y);
    object->SetPos_dt(init_vel);
    object->SetWvel_par(init_omg);
    object->SetCollide(true);
    object->SetBodyFixed(false);

    switch (object->GetContactMethod()) {
        case ChMaterialSurface::NSC:
            object->GetMaterialSurfaceNSC()->SetFriction(object_friction);
            object->GetMaterialSurfaceNSC()->SetRestitution(object_restitution);
            break;
        case ChMaterialSurface::SMC:
            object->GetMaterialSurfaceSMC()->SetFriction(object_friction);
            object->GetMaterialSurfaceSMC()->SetRestitution(object_restitution);
            object->GetMaterialSurfaceSMC()->SetYoungModulus(object_young_modulus);
            object->GetMaterialSurfaceSMC()->SetPoissonRatio(object_poisson_ratio);
            object->GetMaterialSurfaceSMC()->SetKn(object_kn);
            object->GetMaterialSurfaceSMC()->SetGn(object_gn);
            object->GetMaterialSurfaceSMC()->SetKt(object_kt);
            object->GetMaterialSurfaceSMC()->SetGt(object_gt);
            break;
    }

    switch (object_geometry) {
        case GeometryType::MESH: {
            auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
            trimesh->LoadWavefrontMesh(GetChronoDataFile("vehicle/hmmwv/hmmwv_tire_coarse.obj"), true, false);

            object->GetCollisionModel()->ClearModel();
            object->GetCollisionModel()->AddTriangleMesh(trimesh, false, false, ChVector<>(0), ChMatrix33<>(1),
                                                         mesh_swept_sphere_radius);
            object->GetCollisionModel()->BuildModel();

            auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
            trimesh_shape->SetMesh(trimesh);
            trimesh_shape->SetName("tire");
            object->AddAsset(trimesh_shape);

            std::shared_ptr<ChColorAsset> mcol(new ChColorAsset);
            mcol->SetColor(ChColor(0.3f, 0.3f, 0.3f));
            object->AddAsset(mcol);

            break;
        }
        case GeometryType::PRIMITIVE: {
            object->GetCollisionModel()->ClearModel();
            object->GetCollisionModel()->AddSphere(0.2, ChVector<>(0, 0, 0));
            object->GetCollisionModel()->BuildModel();

            auto sphere = chrono_types::make_shared<ChSphereShape>();
            sphere->GetSphereGeometry().rad = 0.2;
            object->AddAsset(sphere);

            break;
        }
    }

    // Create ground body
    auto ground = std::shared_ptr<ChBody>(system->NewBody());
    system->AddBody(ground);

    ground->SetMass(1);
    ground->SetPos(ChVector<>(0, 0, 0));
    ground->SetRot(z2y);
    ground->SetCollide(true);
    ground->SetBodyFixed(true);

    switch (object->GetContactMethod()) {
        case ChMaterialSurface::NSC:
            ground->GetMaterialSurfaceNSC()->SetFriction(ground_friction);
            ground->GetMaterialSurfaceNSC()->SetRestitution(ground_restitution);
            break;
        case ChMaterialSurface::SMC:
            ground->GetMaterialSurfaceSMC()->SetFriction(ground_friction);
            ground->GetMaterialSurfaceSMC()->SetRestitution(ground_restitution);
            ground->GetMaterialSurfaceSMC()->SetYoungModulus(ground_young_modulus);
            ground->GetMaterialSurfaceSMC()->SetPoissonRatio(ground_poisson_ratio);
            ground->GetMaterialSurfaceSMC()->SetKn(ground_kn);
            ground->GetMaterialSurfaceSMC()->SetGn(ground_gn);
            ground->GetMaterialSurfaceSMC()->SetKt(ground_kt);
            ground->GetMaterialSurfaceSMC()->SetGt(ground_gt);
            break;
    }

    switch (ground_geometry) {
        case GeometryType::PRIMITIVE: {
            ground->GetCollisionModel()->ClearModel();
            ground->GetCollisionModel()->AddBox(width, length, thickness, ChVector<>(0, 0, -thickness));
            ground->GetCollisionModel()->BuildModel();

            auto box = chrono_types::make_shared<ChBoxShape>();
            box->GetBoxGeometry().Size = ChVector<>(width, length, thickness);
            ////box->GetBoxGeometry().Pos = ChVector<>(0, 0, -thickness); // not interpreted in parallel collision sys
            box->Pos = ChVector<>(0, 0, -thickness);
            ground->AddAsset(box);

            break;
        }
        case GeometryType::MESH: {
            auto trimesh_ground = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
            trimesh_ground->LoadWavefrontMesh(GetChronoDataFile("vehicle/terrain/meshes/ramp_10x1.obj"), true, false);

            ground->GetCollisionModel()->ClearModel();
            ground->GetCollisionModel()->AddTriangleMesh(trimesh_ground, false, false, ChVector<>(0), ChMatrix33<>(1),
                                                         mesh_swept_sphere_radius);
            ground->GetCollisionModel()->BuildModel();

            auto trimesh_ground_shape = chrono_types::make_shared<ChTriangleMeshShape>();
            trimesh_ground_shape->SetMesh(trimesh_ground);
            trimesh_ground_shape->SetName("ground");
            ground->AddAsset(trimesh_ground_shape);

            break;
        }
    }

#ifdef CHRONO_OPENGL
    // -------------------------------
    // Create the visualization window
    // -------------------------------

    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, "Mesh-mesh test", system);
    gl_window.SetCamera(ChVector<>(2, 1, 2), ChVector<>(0, 0, 0), ChVector<>(0, 1, 0), 0.05f);
    gl_window.SetRenderMode(opengl::WIREFRAME);
#endif

    // ---------------
    // Simulation loop
    // ---------------

    double time_end = 4;
    while (system->GetChTime() < time_end) {
        // Advance dynamics
        system->DoStepDynamics(time_step);

        std::cout << "\nTime: " << system->GetChTime() << std::endl;

#ifdef CHRONO_OPENGL
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        if (gl_window.Active()) {
            gl_window.Render();
        } else {
            return 1;
        }
#endif
    }

    return 0;
}
