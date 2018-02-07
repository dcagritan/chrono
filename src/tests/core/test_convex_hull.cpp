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
//  Test for 2D convex hull
//
// =============================================================================

#include "chrono/core/ChLog.h"
#include "chrono/utils/ChConvexHull.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_thirdparty/filesystem/path.h"
#include "chrono_thirdparty/filesystem/resolver.h"

using namespace chrono;
using namespace filesystem;

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    ////std::vector<ChVector2<>> points = {
    ////        ChVector2<>( -0.22500 , -0.08334   ),    //
    ////        ChVector2<>( -0.24375 , -0.06250   ),    //
    ////        ChVector2<>( -0.20625 , -0.10417   ),    //
    ////        ChVector2<>( -0.26250 , -0.04170   ),    //
    ////        ChVector2<>( -0.26250 , -0.08334   ),    //
    ////        ChVector2<>( -0.24375 , -0.10417   ),    //
    ////        ChVector2<>( -0.22500 , -0.10417   ),    //
    ////        ChVector2<>( -0.22500 , -0.06250   ),    //
    ////        ChVector2<>( -0.26250 , -0.02084   ),    //
    ////        ChVector2<>( -0.26250 , -0.06250   )     //
    ////};

    //// std::vector<ChVector2<>> points = {
    ////     ChVector2<>(0.074999999999999956, -0.33333333333333304),  //
    ////     ChVector2<>(0.14999999999999991, -0.25000000000000000),   //
    ////     ChVector2<>(0.093749999999999944, -0.31249999999999978),  //
    ////     ChVector2<>(0.22499999999999998, -0.25000000000000000),   //
    ////     ChVector2<>(0.26250000000000001, -0.20833333333333326),   //
    ////     ChVector2<>(0.14999999999999991, -0.27083333333333326),   //
    ////     ChVector2<>(0.18749999999999994, -0.25000000000000000),   //
    ////     ChVector2<>(0.18749999999999994, -0.20833333333333326),   //
    ////     ChVector2<>(0.11249999999999993, -0.29166666666666652),   //
    ////     ChVector2<>(0.13124999999999992, -0.27083333333333326),   //
    ////     ChVector2<>(0.16874999999999993, -0.27083333333333326),   //
    ////     ChVector2<>(0.20624999999999996, -0.27083333333333326),   //
    ////     ChVector2<>(0.18749999999999994, -0.27083333333333326),   //
    ////     ChVector2<>(0.22499999999999998, -0.27083333333333326),   //
    ////     ChVector2<>(0.22499999999999998, -0.20833333333333326),   //
    ////     ChVector2<>(0.20624999999999996, -0.22916666666666663),   //
    ////     ChVector2<>(0.16874999999999993, -0.22916666666666663),   //
    ////     ChVector2<>(0.18749999999999994, -0.22916666666666663),   //
    ////     ChVector2<>(0.24374999999999999, -0.22916666666666663),   //
    ////     ChVector2<>(0.22499999999999998, -0.22916666666666663)    //
    ////};

    std::vector<ChVector2<>> points = {
        ChVector2<>(0.074999999999999956, -0.33333333333333304),     //
        ChVector2<>(0.074999999999999956, -0.25000000000000000),     //
        ChVector2<>(0.074999999999999956, -0.31249999999999978),     //
        ChVector2<>(0.22499999999999998, -0.25000000000000000),      //
        ChVector2<>(0.26250000000000001, -0.20833333333333326),      //
        ChVector2<>(0.14999999999999991, -0.27083333333333326),      //
        ChVector2<>(0.18749999999999994, -0.25000000000000000),      //
        ChVector2<>(0.074999999999999956, -0.20833333333333326),     //
        ChVector2<>(0.074999999999999956, -0.29166666666666652),     //
        ChVector2<>(0.074999999999999956, -0.27083333333333326),     //
        ChVector2<>(0.16874999999999993, -0.27083333333333326),      //
        ChVector2<>(0.20624999999999996, -0.27083333333333326),      //
        ChVector2<>(0.18749999999999994, -0.27083333333333326),      //
        ChVector2<>(0.22499999999999998, -0.27083333333333326),      //
        ChVector2<>(0.22499999999999998, -0.20833333333333326),      //
        ChVector2<>(0.20624999999999996, -0.22916666666666663),      //
        ChVector2<>(0.074999999999999956, -0.22916666666666663),     //
        ChVector2<>(0.18749999999999994, -0.22916666666666663),      //
        ChVector2<>(0.24374999999999999, -0.22916666666666663),      //
        ChVector2<>(0.22499999999999998, -0.22916666666666663)       //
    };

    ////std::vector<ChVector2<>> points = {
    ////    ChVector2<>(0.074999999999999956, -0.27083333333333326),     //
    ////    ChVector2<>(0.074999999999999956, -0.25000000000000000),     //
    ////    ChVector2<>(0.074999999999999956, -0.31249999999999978),     //
    ////    ChVector2<>(0.22499999999999998, -0.25000000000000000),      //
    ////    ChVector2<>(0.26250000000000001, -0.20833333333333326),      //
    ////    ChVector2<>(0.14999999999999991, -0.27083333333333326),      //
    ////    ChVector2<>(0.18749999999999994, -0.25000000000000000),      //
    ////    ChVector2<>(0.074999999999999956, -0.20833333333333326),     //
    ////    ChVector2<>(0.074999999999999956, -0.29166666666666652),     //
    ////    ChVector2<>(0.074999999999999956, -0.33333333333333304),     //
    ////    ChVector2<>(0.16874999999999993, -0.27083333333333326),      //
    ////    ChVector2<>(0.20624999999999996, -0.27083333333333326),      //
    ////    ChVector2<>(0.18749999999999994, -0.27083333333333326),      //
    ////    ChVector2<>(0.22499999999999998, -0.27083333333333326),      //
    ////    ChVector2<>(0.22499999999999998, -0.20833333333333326),      //
    ////    ChVector2<>(0.20624999999999996, -0.22916666666666663),      //
    ////    ChVector2<>(0.074999999999999956, -0.22916666666666663),     //
    ////    ChVector2<>(0.18749999999999994, -0.22916666666666663),      //
    ////    ChVector2<>(0.24374999999999999, -0.22916666666666663),      //
    ////    ChVector2<>(0.22499999999999998, -0.22916666666666663)       //
    ////};

    // Jarvis: issue when crt=0
    ////std::vector<ChVector2<>> points = {
    ////    ChVector2<>(-0.18750000000000000, 2.5000000000000000),   //
    ////    ChVector2<>(0.18749999999999994, 2.5000000000000000),    //
    ////    ChVector2<>(-0.22499999999999998, 2.5000000000000000),   //
    ////    ChVector2<>(0.093749999999999944, 2.4791666666666665),   //
    ////    ChVector2<>(-0.15000000000000002, 2.5000000000000000),   //
    ////    ChVector2<>(0.11249999999999993, 2.5000000000000000),    //
    ////    ChVector2<>(-0.075000000000000067, 2.5000000000000000),  //
    ////    ChVector2<>(0.00000000000000000, 2.5000000000000000),    //
    ////    ChVector2<>(0.14999999999999991, 2.5000000000000000),    //
    ////    ChVector2<>(0.22499999999999998, 2.5000000000000000),    //
    ////    ChVector2<>(0.037499999999999978, 2.5000000000000000),   //
    ////    ChVector2<>(0.018749999999999989, 2.4791666666666665),   //
    ////    ChVector2<>(0.16874999999999993, 2.4791666666666665),    //
    ////    ChVector2<>(-0.11250000000000004, 2.5000000000000000),   //
    ////    ChVector2<>(-0.093750000000000056, 2.4791666666666665),  //
    ////    ChVector2<>(-0.037500000000000033, 2.5000000000000000),  //
    ////    ChVector2<>(-0.056250000000000050, 2.4791666666666665),  //
    ////    ChVector2<>(-0.075000000000000067, 2.4791666666666665),  //
    ////    ChVector2<>(-0.018750000000000017, 2.4791666666666665),  //
    ////    ChVector2<>(-0.037500000000000033, 2.4791666666666665),  //
    ////    ChVector2<>(0.00000000000000000, 2.4791666666666665),    //
    ////    ChVector2<>(0.056249999999999967, 2.4791666666666665),   //
    ////    ChVector2<>(0.037499999999999978, 2.4791666666666665),   //
    ////    ChVector2<>(0.074999999999999956, 2.4791666666666665),   //
    ////    ChVector2<>(0.13124999999999992, 2.4791666666666665),    //
    ////    ChVector2<>(0.14999999999999991, 2.4791666666666665),    //
    ////    ChVector2<>(0.11249999999999993, 2.4791666666666665)     //
    ////};

    ////std::vector<ChVector2<>> points = { ChVector2<>(0.18749999999999994, -0.41666666666666652),
    ////    ChVector2<>(0.20624999999999996, -0.39583333333333315),
    ////    ChVector2<>(0.16874999999999993, -0.43749999999999989) };

    std::string out_dir = "../TEST_convex_hull";
    bool out_dir_exists = path(out_dir).exists();
    if (out_dir_exists) {
        cout << "Output directory already exists" << endl;
    } else if (create_directory(path(out_dir))) {
        cout << "Create directory = " << path(out_dir).make_absolute() << endl;
    } else {
        cout << "Error creating output directory" << endl;
        return 1;
    }
    utils::CSV_writer csv("\t");
    csv.stream().setf(std::ios::scientific | std::ios::showpos);
    csv.stream().precision(6);

    utils::ChConvexHull2D ch(points);
    auto hull = ch.GetHull();
    for (auto p : hull) {
        std::cout << p.x() << " " << p.y() << std::endl;
        csv << p.x() << p.y() << std::endl;
    }

    csv.write_to_file(out_dir + "/hull.txt");

    return 0;
}
