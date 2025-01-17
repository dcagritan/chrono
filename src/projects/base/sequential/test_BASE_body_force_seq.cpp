#include <omp.h>
#include <cstdio>
#include <vector>
#include <cmath>

#include "chrono/ChConfig.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChLoadsBody.h"

#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/utils/ChUtilsCreators.h"

#include "chrono_irrlicht/ChVisualSystemIrrlicht.h"
#include "chrono_irrlicht/ChIrrTools.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::collision;

int main(int argc, char* argv[]) {
    double time_step = 1e-3;
    double time_end = 5;
    double time_start = 1;

    double out_fps = 50;

    double tolerance = 1e-3;
    unsigned int max_iteration = 100;

    enum ForceType { BODY_FORCE, LOAD_FORCE, ACCUMULATOR_FORCE };
    ForceType force_type = ACCUMULATOR_FORCE;

    bool floating = false;

    float friction = 0.5f;

    // Create system
    ChSystemSMC sys;
    ////ChSystemNSC sys;

    if (floating)
        sys.Set_G_acc(ChVector<double>(0, 0, 0));
    else
        sys.Set_G_acc(ChVector<double>(0, 0, -4));

    // Create the plate
    auto plate = std::shared_ptr<ChBody>(sys.NewBody());

    plate->SetIdentifier(0);
    plate->SetPos(ChVector<>(0, 0, 0));
    plate->SetBodyFixed(true);
    plate->SetCollide(true);

    plate->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(plate.get(), ChMaterialSurface::DefaultMaterial(sys.GetContactMethod()),
                          ChVector<>(8, 1, 0.2));
    plate->GetCollisionModel()->BuildModel();

    sys.AddBody(plate);

    // Create the ball
    double mass = 1;
    double radius = 0.15;
    ChVector<> inertia = (2.0 / 5.0) * mass * radius * radius * ChVector<>(1, 1, 1);

    auto ball = std::shared_ptr<ChBody>(sys.NewBody());
    ball->SetIdentifier(2);
    ball->SetMass(mass);
    ball->SetInertiaXX(inertia);
    ball->SetPos(ChVector<>(-3, 0, 1));
    ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ball->SetBodyFixed(false);
    ball->SetCollide(true);

    auto ball_mat = ChMaterialSurface::DefaultMaterial(sys.GetContactMethod());
    ball_mat->SetFriction(friction);
    ball_mat->SetRestitution(0);

    ball->GetCollisionModel()->ClearModel();
    utils::AddSphereGeometry(ball.get(), ball_mat, radius);
    ball->GetCollisionModel()->BuildModel();

    ball->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/greenwhite.png"));

    sys.AddBody(ball);

    // Force (ramp to maximum and then back to 0)
    double fullForce = 1;
    auto force_function = chrono_types::make_shared<ChFunction_Recorder>();
    force_function->AddPoint(0, 0);
    force_function->AddPoint(time_start, 0);
    force_function->AddPoint(time_start + 1, fullForce);
    force_function->AddPoint(time_start + 2, fullForce);
    force_function->AddPoint(time_start + 3, 0);
    force_function->AddPoint(time_end, 0);

    // Torque
    auto torque_function = chrono_types::make_shared<ChFunction_Sine>(0.0, 1.0, 1.0);

    auto body_force = chrono_types::make_shared<ChForce>();
    auto body_torque = chrono_types::make_shared<ChForce>();
    auto load_force = chrono_types::make_shared<ChLoadBodyForce>(ball, VNULL, false, VNULL, false);
    auto load_torque = chrono_types::make_shared<ChLoadBodyTorque>(ball, VNULL, false);

    if (force_type == BODY_FORCE) {
        ball->AddForce(body_force);
        body_force->SetMode(ChForce::ForceType::FORCE);
        body_force->SetFrame(ChForce::ReferenceFrame::BODY);
        body_force->SetAlign(ChForce::AlignmentFrame::WORLD_DIR);
        body_force->SetVrelpoint(ChVector<>(0, 0, 0));
        body_force->SetF_x(force_function);
        body_force->SetF_y(chrono_types::make_shared<ChFunction_Const>(0));
        body_force->SetF_z(chrono_types::make_shared<ChFunction_Const>(0));

        ball->AddForce(body_torque);
        body_torque->SetMode(ChForce::ForceType::TORQUE);
        body_torque->SetFrame(ChForce::ReferenceFrame::BODY);
        body_torque->SetAlign(ChForce::AlignmentFrame::WORLD_DIR);
        body_torque->SetVrelpoint(ChVector<>(0, 0, 0));
        body_torque->SetF_x(chrono_types::make_shared<ChFunction_Const>(0));
        body_torque->SetF_y(torque_function);
        body_torque->SetF_z(chrono_types::make_shared<ChFunction_Const>(0));
    } else if (force_type == LOAD_FORCE) {
        auto load_container = chrono_types::make_shared<ChLoadContainer>();
        load_container->Add(load_force);
        load_container->Add(load_torque);
        sys.Add(load_container);
    }

    const std::string out_dir = GetChronoOutputPath() + "TEST_BODY_FORCE";
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cout << "Error creating directory " << out_dir << std::endl;
        return 1;
    }

    utils::CSV_writer csv(" ");

    // Create the visualization window
    auto vis = chrono_types::make_shared<irrlicht::ChVisualSystemIrrlicht>();
    vis->SetCameraVertical(CameraVerticalDir::Z);
    vis->SetWindowSize(800, 600);
    vis->SetWindowTitle("Body force test");
    vis->Initialize();
    vis->AddLogo();
    vis->AddSkyBox();
    vis->AddTypicalLights();
    vis->AddCamera(ChVector<>(0, -4, 2));
    vis->AttachSystem(&sys);

    // Run simulation for specified time
    while (true) {
        double time = sys.GetChTime();

        if (time > time_end)
            break;
        if (vis->Run()) {
            vis->BeginScene();
            vis->Render();
            irrlicht::tools::drawAllCOGs(vis.get(), 1);
            vis->EndScene();
        } else {
            break;
        }

        double Fx = force_function->Get_y(time);
        double Ty = torque_function->Get_y(time);

        if (force_type == ACCUMULATOR_FORCE) {
            ball->Empty_forces_accumulators();
            ball->Accumulate_force(ChVector<>(Fx, 0, 0), ball->GetPos(), false);
            ball->Accumulate_torque(ChVector<>(0, Ty, 0), false);
        } else if (force_type == LOAD_FORCE) {
            load_force->SetForce(ChVector<>(Fx, 0, 0), false);
            load_force->SetApplicationPoint(ball->GetPos(), false);
            load_torque->SetTorque(ChVector<>(0, Ty, 0), false);
        }

        sys.DoStepDynamics(time_step);

        csv << time << ball->GetPos() << ball->GetRot() << ball->GetPos_dt() << ball->GetWvel_par() << std::endl;
    }

    switch (force_type) {
        case ACCUMULATOR_FORCE:
            csv.write_to_file(out_dir + "/Accumulator_force.txt");
            break;
        case BODY_FORCE:
            csv.write_to_file(out_dir + "/Body_force.txt");
            break;
        case LOAD_FORCE:
            csv.write_to_file(out_dir + "/Load_force.txt");
            break;
    }

    return 0;
}
