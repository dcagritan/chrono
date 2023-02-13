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
#include "chrono/assets/ChBoxShape.h"
#include "chrono/assets/ChSphereShape.h"

#include "chrono_multicore/physics/ChSystemMulticore.h"
#include "chrono_multicore/solver/ChIterativeSolverMulticore.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChVisualSystemOpenGL.h"
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

    ChContactMethod contact_method = ChContactMethod::NSC;
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

    ChSystemMulticore* system;

    switch (contact_method) {
        case ChContactMethod::NSC: {
            auto my_sys = new ChSystemMulticoreNSC();
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
        case ChContactMethod::SMC: {
            auto my_sys = new ChSystemMulticoreSMC();
            my_sys->GetSettings()->solver.contact_force_model = ChSystemSMC::Hertz;
            my_sys->GetSettings()->solver.tangential_displ_mode = ChSystemSMC::OneStep;

            system = my_sys;
            break;
        }
    }

    system->Set_G_acc(ChVector<>(0, -gravity, 0));

    int nthreads = 2;
    int max_threads = omp_get_num_procs();
    if (nthreads > max_threads)
        nthreads = max_threads;
    system->SetNumThreads(nthreads);

    system->GetSettings()->solver.use_full_inertia_tensor = false;
    system->GetSettings()->solver.tolerance = tolerance;

    system->GetSettings()->collision.collision_envelope = collision_envelope;
    system->GetSettings()->collision.narrowphase_algorithm = collision::ChNarrowphase::Algorithm::HYBRID;
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

    auto object_mat = ChMaterialSurface::DefaultMaterial(contact_method);
    object_mat->SetFriction(object_friction);
    object_mat->SetRestitution(object_restitution);
    if (contact_method == ChContactMethod::SMC) {
        auto matSMC = std::static_pointer_cast<ChMaterialSurfaceSMC>(object_mat);
        matSMC->SetYoungModulus(object_young_modulus);
        matSMC->SetPoissonRatio(object_poisson_ratio);
        matSMC->SetKn(object_kn);
        matSMC->SetGn(object_gn);
        matSMC->SetKt(object_kt);
        matSMC->SetGt(object_gt);
    }

    switch (object_geometry) {
        case GeometryType::MESH: {
            auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
            trimesh->LoadWavefrontMesh(GetChronoDataFile("vehicle/hmmwv/hmmwv_tire_coarse.obj"), true, false);

            object->GetCollisionModel()->ClearModel();
            object->GetCollisionModel()->AddTriangleMesh(object_mat, trimesh, false, false, ChVector<>(0), ChMatrix33<>(1),
                                                         mesh_swept_sphere_radius);
            object->GetCollisionModel()->BuildModel();

            auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
            trimesh_shape->SetMesh(trimesh);
            trimesh_shape->SetName("tire");
            trimesh_shape->SetColor(ChColor(0.3f, 0.3f, 0.3f));
            object->AddVisualShape(trimesh_shape);

            break;
        }
        case GeometryType::PRIMITIVE: {
            object->GetCollisionModel()->ClearModel();
            object->GetCollisionModel()->AddSphere(object_mat, 0.2, ChVector<>(0, 0, 0));
            object->GetCollisionModel()->BuildModel();

            auto sphere = chrono_types::make_shared<ChSphereShape>();
            sphere->GetSphereGeometry().rad = 0.2;
            object->AddVisualShape(sphere);

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

    auto ground_mat = ChMaterialSurface::DefaultMaterial(contact_method);
    ground_mat->SetFriction(ground_friction);
    ground_mat->SetRestitution(ground_restitution);
    if (contact_method == ChContactMethod::SMC) {
        auto matSMC = std::static_pointer_cast<ChMaterialSurfaceSMC>(ground_mat);
        matSMC->SetYoungModulus(ground_young_modulus);
        matSMC->SetPoissonRatio(ground_poisson_ratio);
        matSMC->SetKn(ground_kn);
        matSMC->SetGn(ground_gn);
        matSMC->SetKt(ground_kt);
        matSMC->SetGt(ground_gt);
    }

    switch (ground_geometry) {
        case GeometryType::PRIMITIVE: {
            ground->GetCollisionModel()->ClearModel();
            ground->GetCollisionModel()->AddBox(ground_mat, width, length, thickness, ChVector<>(0, 0, -thickness));
            ground->GetCollisionModel()->BuildModel();

            auto box = chrono_types::make_shared<ChBoxShape>();
            box->GetBoxGeometry().Size = ChVector<>(width, length, thickness);
            ground->AddVisualShape(box, ChFrame<>(ChVector<>(0, 0, -thickness)));

            break;
        }
        case GeometryType::MESH: {
            auto trimesh_ground = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
            trimesh_ground->LoadWavefrontMesh(GetChronoDataFile("vehicle/terrain/meshes/ramp_10x1.obj"), true, false);

            ground->GetCollisionModel()->ClearModel();
            ground->GetCollisionModel()->AddTriangleMesh(ground_mat, trimesh_ground, false, false, ChVector<>(0), ChMatrix33<>(1),
                                                         mesh_swept_sphere_radius);
            ground->GetCollisionModel()->BuildModel();

            auto trimesh_ground_shape = chrono_types::make_shared<ChTriangleMeshShape>();
            trimesh_ground_shape->SetMesh(trimesh_ground);
            trimesh_ground_shape->SetName("ground");
            ground->AddVisualShape(trimesh_ground_shape);

            break;
        }
    }

#ifdef CHRONO_OPENGL
    // -------------------------------
    // Create the visualization window
    // -------------------------------

    opengl::ChVisualSystemOpenGL vis;
    vis.AttachSystem(system);
    vis.SetWindowTitle("Test");
    vis.SetWindowSize(1280, 720);
    vis.SetRenderMode(opengl::WIREFRAME);
    vis.Initialize();
    vis.AddCamera(ChVector<>(2, 1, 2), ChVector<>(0, 0, 0));
    vis.SetCameraVertical(CameraVerticalDir::Z);

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
        if (vis.Run()) {
            vis.Render();
        } else {
            return 1;
        }
#endif
    }

    return 0;
}
