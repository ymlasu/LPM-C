#pragma once
#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "lpm.h"

struct dispBCPara;
struct forceBCPara;

void setDispBC_stiffnessUpdate2D();
void setDispBC_stiffnessUpdate3D();

void setDispBC(int nboundDisp, struct dispBCPara* dBP);
void setForceBC(int nboundForce, struct forceBCPara* fBP);

#endif
