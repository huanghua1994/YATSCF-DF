#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"

// If BLOCK_SIZE is too small, the flop-per-byte ratio is small for dgemm and cannot have good 
// performance. The redundant calculation in K matrix build is BLOCK_SIZE / nbf * 100%.
// Consider using a dynamic value later. 
#define BLOCK_SIZE 32

static int BLOCK_LOW(int i, int n, int size)
{
	int remainder = size % n;
	int bs0 = size / n;
	int bs1 = bs0 + 1;
	int res;
	if (i <= remainder) res = bs1 * i;
	else res = bs0 * i + remainder;
	return res;
}

// Generate temporary array for J matrix and form J matrix
// Low flop-per-byte ratio: access: nbf^2 * (df_nbf+1), compute: nbf^2 * df_nbf 
static void build_J_mat(TinySCF_t TinySCF)
{
	double *J_mat     = TinySCF->J_mat;
	double *D_mat     = TinySCF->D_mat;
	double *df_tensor = TinySCF->df_tensor;
	double *temp_J    = TinySCF->temp_J;
	int nbf           = TinySCF->nbasfuncs;
	int df_nbf        = TinySCF->df_nbf;
	int nthreads      = TinySCF->nthreads;
	
	#pragma omp parallel
	{
		int tid  = omp_get_thread_num();
		int spos = BLOCK_LOW(tid, nthreads, df_nbf);
		int epos = BLOCK_LOW(tid + 1, nthreads, df_nbf);

		for (int i = spos; i < epos; i++) temp_J[i] = 0;

		// Generate temporary array for J
		for (int k = 0; k < nbf; k++)
		{
			for (int l = 0; l < nbf; l++)
			{
				double D_kl = D_mat[k * nbf + l];
				double *df_tensor_row = df_tensor + (l * nbf + k) * df_nbf;

				#pragma simd
				for (int p = spos; p < epos; p++)
					temp_J[p] += D_kl * df_tensor_row[p];
			}
		}

		#pragma omp barrier

		// Build J matrix
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < nbf; i++)
		{
			for (int j = i; j < nbf; j++)
			{
				double t = 0;
				double *df_tensor_row = df_tensor + (i * nbf + j) * df_nbf;
				#pragma simd
				for (int p = 0; p < df_nbf; p++)
					t += temp_J[p] * df_tensor_row[p];
				J_mat[i * nbf + j] = t;
			}
		}
	}
}

// Generate the temporary tensor for K matrix and form K matrix
// High flop-per-byte ratio: access: nbf^2 * (2*df_nbf+1), compute: nbf^3 * df_nbf
// Note: the K_mat is not completed, the symmetrizing is done later
void build_K_mat(TinySCF_t TinySCF)
{
	double *K_mat     = TinySCF->K_mat;
	double *D_mat     = TinySCF->D_mat;
	double *df_tensor = TinySCF->df_tensor;
	double *temp_K    = TinySCF->temp_K;
	int nbf           = TinySCF->nbasfuncs;
	int df_nbf        = TinySCF->df_nbf;
	
	// Construct temporary tensor for K matrix
	// Formula: temp_K(k, j, p) = dot(D_mat(k, 1:nbf), df_tensor(1:nbf, j, p))
	for (int j = 0; j < nbf; j++)
	{
		double *temp_K_j    = temp_K    + j * df_nbf;
		double *df_tensor_j = df_tensor + j * df_nbf;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf, df_nbf, nbf, 
		            1.0, D_mat, nbf, df_tensor_j, nbf * df_nbf, 0.0, temp_K_j, nbf * df_nbf);
	}

	// Build K matrix
	// Formula: K(i, j) = sum_{k=1}^{nbf} [ dot(df_tensor(i, k, 1:df_nbf), temp_K(k, j, 1:df_nbf)) ]
	memset(K_mat, 0, DBL_SIZE * nbf * nbf);
	for (int i = 0; i < nbf; i += BLOCK_SIZE)
	{
		int i_len = BLOCK_SIZE < (nbf - i) ? BLOCK_SIZE : (nbf - i);
		for (int j = i; j < nbf; j += BLOCK_SIZE)
		{
			int j_len = BLOCK_SIZE < (nbf - j) ? BLOCK_SIZE : (nbf - j);
			double *K_ij = K_mat + i * nbf + j;
			for (int k = 0; k < nbf; k++)
			{
				double *df_tensor_i = df_tensor + (i * nbf + k) * df_nbf;
				double *temp_K_j = temp_K + (k * nbf + j) * df_nbf;
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_len, j_len, df_nbf,
				            1.0, df_tensor_i, nbf * df_nbf, temp_K_j, df_nbf, 1.0, K_ij, nbf);
			}
		}
	}
}

void TinySCF_build_FockMat(TinySCF_t TinySCF)
{
	double *Hcore_mat = TinySCF->Hcore_mat;
	double *J_mat = TinySCF->J_mat;
	double *K_mat = TinySCF->K_mat;
	double *F_mat = TinySCF->F_mat;
	int nbf = TinySCF->nbasfuncs;
	
	double st0, st1, st2, et0, et1, et2;
	
	st0 = get_wtime_sec();
	
	// Build J matrix
	st1 = get_wtime_sec();
	build_J_mat(TinySCF);
	et1 = get_wtime_sec();
	
	// Build K matrix
	st2 = get_wtime_sec();
	build_K_mat(TinySCF);
	et2 = get_wtime_sec();
	
	// Symmetrizing J and K matrix and build Fock matrix
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic)
		for (int i = 1; i < nbf; i++)
		{
			for (int j = 0; j < i; j++)
			{
				J_mat[i * nbf + j] = J_mat[j * nbf + i];
				K_mat[i * nbf + j] = K_mat[j * nbf + i];
			}
		}
		
		#pragma omp for 
		for (int i = 0; i < nbf * nbf; i++)
			F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] - K_mat[i];
	}
	
	et0 = get_wtime_sec();
	printf("* Fock matrix build     : %.3lf (s)\n", et0 - st0);
	printf("|--- J matrix build     |- %.3lf (s)\n", et1 - st1);
	printf("|--- K matrix build     |- %.3lf (s)\n", et2 - st2);
}