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
// Authors: Radu Serban
// =============================================================================
//
// Chrono::Distributed test program using SMC method for frictional contact.
//
// The model simulated here consists of a number of objects falling in a bin.
//
// The global reference frame has Z up.
// =============================================================================

#include <cmath>
#include <cstdio>

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_multicore/physics/ChSystemMulticore.h"

#include "chrono_distributed/collision/ChBoundary.h"

#include "chrono_opengl/ChVisualSystemOpenGL.h"

using namespace chrono;
using namespace chrono::collision;

// Container collision model
enum ContainerType { BOXES, BOUNDARY };
ContainerType container_type = BOUNDARY;

// Tilt angle (about global Y axis) of the container.
double tilt_angle = 1 * CH_C_PI / 20;

// Shape of falling objects
enum ObjectType { BALL, BRICK };
ObjectType object_type = BALL;

// Number of objects: (2 * count_X + 1) * (2 * count_Y + 1) * count_Z
int count_X = 2;
int count_Y = 2;
int count_Z = 2;

// Material properties (same on bin and objects)
float Y = 2e6f;
float mu = 0.4f;
float cr = 0.4f;

// -----------------------------------------------------------------------------
// Create a bin attached to the ground.
// -----------------------------------------------------------------------------
void AddContainer(ChSystemMulticoreSMC* sys) {
    // IDs for the two bodies
    int binId = -200;

    // Create a common material
    auto mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    mat->SetYoungModulus(Y);
    mat->SetFriction(mu);
    mat->SetRestitution(cr);

    // Create the containing bin (4 x 4 x 1)
    auto bin = chrono_types::make_shared<ChBody>(chrono_types::make_shared<ChCollisionModelChrono>());
    bin->SetIdentifier(binId);
    bin->SetMass(1);
    bin->SetPos(ChVector<>(0, 0, 0));
    bin->SetRot(Q_from_AngY(tilt_angle));
    bin->SetCollide(true);
    bin->SetBodyFixed(true);
    sys->AddBody(bin);

    ChVector<> hdim(2, 2, 0.5);
    double hthick = 0.1;

    switch (container_type) {
        case BOXES: {
            bin->GetCollisionModel()->ClearModel();
            utils::AddBoxGeometry(bin.get(), mat, ChVector<>(hdim.x(), hdim.y(), hthick), ChVector<>(0, 0, -hthick));
            utils::AddBoxGeometry(bin.get(), mat, ChVector<>(hthick, hdim.y(), hdim.z()),
                                  ChVector<>(-hdim.x() - hthick, 0, hdim.z()));
            utils::AddBoxGeometry(bin.get(), mat, ChVector<>(hthick, hdim.y(), hdim.z()),
                                  ChVector<>(hdim.x() + hthick, 0, hdim.z()));
            utils::AddBoxGeometry(bin.get(), mat, ChVector<>(hdim.x(), hthick, hdim.z()),
                                  ChVector<>(0, -hdim.y() - hthick, hdim.z()));
            utils::AddBoxGeometry(bin.get(), mat, ChVector<>(hdim.x(), hthick, hdim.z()),
                                  ChVector<>(0, hdim.y() + hthick, hdim.z()));
            bin->GetCollisionModel()->BuildModel();

            break;
        }
        case BOUNDARY: {
            auto cb = new ChBoundary(bin, mat);
            cb->AddPlane(ChFrame<>(ChVector<>(0, 0, 0), QUNIT), ChVector2<>(4, 4));
            cb->AddPlane(ChFrame<>(ChVector<>(2, 0, 0), Q_from_AngY(-CH_C_PI_2)), ChVector2<>(1, 4));
            cb->AddVisualization(0.05);

            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Create the falling objects in a uniform rectangular grid.
// Return a handle to the first created object.
// -----------------------------------------------------------------------------
std::shared_ptr<ChBody> AddFallingObjects(ChSystemMulticore* sys) {
    std::shared_ptr<ChBody> first;

    // Common material
    auto objMat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    objMat->SetYoungModulus(Y);
    objMat->SetFriction(mu);
    objMat->SetRestitution(cr);
    objMat->SetAdhesion(0);

    // Create the falling objects
    int objId = 0;
    double mass = 1;
    double size = 0.15;
    ChVector<> inertia = (2.0 / 5.0) * mass * size * size * ChVector<>(1, 1, 1);

    for (int iz = 0; iz < count_Z; iz++) {
        double h = 1 + 0.4 * iz;
        for (int ix = -count_X; ix <= count_X; ix++) {
            for (int iy = -count_Y; iy <= count_Y; iy++) {
                ChVector<> pos(0.4 * ix, 0.4 * iy, h);

                auto obj = chrono_types::make_shared<ChBody>(chrono_types::make_shared<ChCollisionModelChrono>());

                obj->SetIdentifier(objId++);
                obj->SetMass(mass);
                obj->SetInertiaXX(inertia);
                obj->SetPos(pos);
                obj->SetRot(ChQuaternion<>(1, 0, 0, 0));
                obj->SetBodyFixed(false);
                obj->SetCollide(true);

                obj->GetCollisionModel()->ClearModel();
                switch (object_type) {
                case BALL:
                    utils::AddSphereGeometry(obj.get(), objMat, size);
                    break;
                case BRICK:
                    utils::AddBoxGeometry(obj.get(), objMat, ChVector<>(size, size / 2, size / 2));
                    break;
                }
                obj->GetCollisionModel()->BuildModel();

                sys->AddBody(obj);

                if (!first)
                    first = obj;
            }
        }
    }

    return first;
}

// -----------------------------------------------------------------------------
// Create the system, specify simulation parameters, and run simulation loop.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    int threads = 8;

    // Simulation parameters
    // ---------------------

    double gravity = 9.81;
    double time_step = 1e-3;

    uint max_iteration = 100;
    real tolerance = 1e-3;

    // Create system
    // -------------

    ChSystemMulticoreSMC msystem;

    // Set number of threads.
    int max_threads = ChOMP::GetNumProcs();
    if (threads > max_threads)
        threads = max_threads;
    msystem.SetNumThreads(threads);

    // Set gravitational acceleration
    msystem.Set_G_acc(ChVector<>(0, 0, -gravity));

    // Set solver parameters
    msystem.GetSettings()->solver.max_iteration_bilateral = max_iteration;
    msystem.GetSettings()->solver.tolerance = tolerance;

    msystem.GetSettings()->collision.narrowphase_algorithm = ChNarrowphase::Algorithm::MPR;
    msystem.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    msystem.GetSettings()->solver.contact_force_model = ChSystemSMC::ContactForceModel::Hooke;
    msystem.GetSettings()->solver.adhesion_force_model = ChSystemSMC::AdhesionForceModel::Constant;

    // Create the fixed and moving bodies
    // ----------------------------------
    AddContainer(&msystem);
    auto object = AddFallingObjects(&msystem);

    // Perform the simulation
    // ----------------------

    opengl::ChVisualSystemOpenGL vis;
    vis.AttachSystem(&msystem);
    vis.SetWindowTitle("Test");
    vis.SetWindowSize(1280, 720);
    vis.SetRenderMode(opengl::WIREFRAME);
    vis.Initialize();
    vis.AddCamera(ChVector<>(0, -6, 0), ChVector<>(0, 0, 0));
    vis.SetCameraVertical(CameraVerticalDir::Z);

    // Uncomment the following two lines for the OpenGL manager to automatically
    // run the simulation in an infinite loop.
    // vis.StartDrawLoop(time_step);
    // return 0;

    while (vis.Run()) {
        msystem.DoStepDynamics(time_step);
        vis.Render();
        ChVector<> pos = object->GetPos();
        real3 frc = msystem.GetBodyContactForce(0);
        std::cout << msystem.GetChTime() << "   " << pos.x() << " " << pos.y() << " " << pos.z() << "    " << frc.x
                  << "  " << frc.y << "  " << frc.z << std::endl;
    }

    return 0;
}
