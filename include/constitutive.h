#pragma once
#ifndef PLASTICITY_H
#define PLASTICITY_H

/* below functions are used for distortional energy criterion */
//void calcPl();
//void updatePl();

void switchStateV(int conv_flag);

void computeCab();
void computeddLp(double **xddLp, double **xcp_gamma);

void computeBondForceGeneral(int plmode, int temp);
void computeBondForceElastic(int i);
void computeBondForceJ2mixedLinear3D(int ii);
void computeBondForceJ2nonlinearIso(int ii);
void computeBondForceCPMiehe(int ii);
void computeBondForceIncrementalUpdating(int ii);
void computeBondForceJ2energyReturnMap(int ii, int load_indicator);

int updateDamageGeneral(const char *dataName, int tstep, int plmode);
int updateBrittleDamage(const char *dataName, int tstep, int nbreak);
int updateDuctileDamageBwiseLocal(const char *dataName, int tstep);
int updateDuctileDamagePwiseLocal(const char *dataName, int tstep);
int updateDuctileDamageBwiseNonlocal(const char *dataName, int tstep);
int updateDuctileDamagePwiseNonlocal(const char *dataName, int tstep);

void updateCrack();

struct bondStrain
{
    int i_index;
    int j_index;
    double bstrain;
};

#endif
