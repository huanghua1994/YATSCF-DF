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
				size_t offset = (size_t) (l * nbf + k) * (size_t) df_nbf;
				double *df_tensor_row = df_tensor + offset;

				#pragma simd
				for (size_t p = spos; p < epos; p++)
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
				size_t offset = (size_t) (i * nbf + j) * (size_t) df_nbf;
				double *df_tensor_row = df_tensor + offset;
				#pragma simd
				for (size_t p = 0; p < df_nbf; p++)
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
	double *K_mat = TinySCF->K_mat;
	int nbf       = TinySCF->nbasfuncs;
	int df_nbf    = TinySCF->df_nbf;
	int n_occ     = TinySCF->n_occ;
	
	// Construct temporary tensor for K matrix
	// Formula: temp_K(s, i, p) = dot(Cocc_mat(1:nbf, s), df_tensor(i, 1:nbf, p))
	const int group_size = nbf;
	const CBLAS_TRANSPOSE temp_K_transa = CblasTrans;
	const CBLAS_TRANSPOSE temp_K_transb = CblasNoTrans;
	const int temp_K_m = n_occ, temp_K_n = df_nbf, temp_K_k = nbf;
	const double temp_K_alpha = 1.0, temp_K_beta = 0.0;
	const int temp_K_lda = n_occ;
	const int temp_K_ldb = df_nbf;
	const int temp_K_ldc = df_nbf * nbf;
	cblas_dgemm_batch(
		CblasRowMajor, &temp_K_transa, &temp_K_transb, 
		&temp_K_m, &temp_K_n, &temp_K_k,
		&temp_K_alpha,
		TinySCF->temp_K_a, &temp_K_lda,
		TinySCF->temp_K_b, &temp_K_ldb,
		&temp_K_beta,
		TinySCF->temp_K_c, &temp_K_ldc,
		1, &group_size
	);

	// Build K matrix
	// Formula: K(i, j) = sum_{s=1}^{n_occ} [ dot(temp_K(s, i, 1:df_nbf), temp_K(s, j, 1:df_nbf)) ]
	memset(K_mat, 0, DBL_SIZE * nbf * nbf);
	for (int s = 0; s < n_occ; s++)
	{
		cblas_dgemm_batch(
			CblasRowMajor, TinySCF->mat_K_transa, TinySCF->mat_K_transb, 
			TinySCF->mat_K_m, TinySCF->mat_K_n, TinySCF->mat_K_k, 
			TinySCF->mat_K_alpha, 
			TinySCF->mat_K_a, TinySCF->mat_K_lda,
			TinySCF->mat_K_b, TinySCF->mat_K_ldb,
			TinySCF->mat_K_beta,
			TinySCF->mat_K_c, TinySCF->mat_K_ldc,
			3, TinySCF->mat_K_group_size
		);
		
		size_t offset = (size_t) df_nbf * (size_t) nbf;
		for (int i = 0; i < TinySCF->mat_K_ntiles; i++)
		{
			TinySCF->mat_K_a[i] += offset;
			TinySCF->mat_K_b[i] += offset;
		}
	}
	// Reset array points to initial values
	size_t offset = (size_t) n_occ * (size_t) df_nbf * (size_t) nbf;
	for (int i = 0; i < TinySCF->mat_K_ntiles; i++)
	{
		TinySCF->mat_K_a[i] -= offset;
		TinySCF->mat_K_b[i] -= offset;
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