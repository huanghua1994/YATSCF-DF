#ifndef _YATSCF_TFOCK_INIT_FREE_
#define _YATSCF_TFOCK_INIT_FREE_

#include "TinySCF.h"

// Initialize TinySCF with two Cartesian basis set file (.gbs format), a molecule 
// coordinate file and the number of SCF iterations (handled by libcint), and
// allocate all memory for other calculation
void init_TinySCF(
	TinySCF_t TinySCF, char *bas_fname, char *df_bas_fname, 
	char *xyz_fname, const int niters
);

// Destroy TinySCF, free all allocated memory
void free_TinySCF(TinySCF_t TinySCF);

// Allocate arrays for MKL batch gemm and precompute tile info
// for building the K matrix
void TinySCF_init_batch_dgemm_arrays(TinySCF_t TinySCF);

// Free MKL batch gemm arrays
void TinySCF_free_batch_dgemm_arrays(TinySCF_t TinySCF);


#endif
