#pragma once
#include "common.h"

Eigen::Matrix<double, 12, 12> L_L2d_ind_m();
Eigen::Matrix<double, 7, 12> L_d2L_dep_m();
Eigen::Matrix<double, 5, 1> c_e_m();