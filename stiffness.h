#pragma once
#ifndef STIFFNESS_H
#define STIFFNESS_H

void calcKnTv();
void updateRR();
void calcStiffness2DFiniteDifference(int plmode);
void calcStiffness3DFiniteDifference(int plmode);

#endif
