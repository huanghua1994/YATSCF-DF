#ifndef _YATSCF_BUILD_DENSITY_H_
#define _YATSCF_BUILD_DENSITY_H_

#include "TinySCF.h"

void TinySCF_build_DenMat(TinySCF_t TinySCF);

void quickSort_eigval(double *eigval, int *ev_idx, int l, int r);

#endif
