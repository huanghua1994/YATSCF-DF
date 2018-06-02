#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <x86intrin.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"

void reduce_temp_J(double *temp_J, double *temp_J_thread, int len, int tid, int nthreads)
{
	while (nthreads > 1)
	{
		int mid = (nthreads + 1) / 2;
		int act_mid = nthreads / 2;
		if (tid < act_mid)
		{
			double *dst = temp_J_thread + len * mid;

			#pragma simd
			for (int i = 0; i < len; i++)
				temp_J_thread[i] += dst[i];
		}

		#pragma omp barrier
		nthreads = mid;
	}
}


// Generate temporary array for J matrix and form J matrix
// Low flop-per-byte ratio: access: nbf^2 * (df_nbf+1), compute: nbf^2 * df_nbf 
static void build_J_mat(TinySCF_t TinySCF, double *temp_J_t, double *J_mat_t)
{
	double *J_mat     = TinySCF->J_mat;
	double *D_mat     = TinySCF->D_mat;
	double *df_tensor = TinySCF->df_tensor;
	double *temp_J    = TinySCF->temp_J;
	int nbf           = TinySCF->nbasfuncs;
	int df_nbf        = TinySCF->df_nbf;
	int nthreads      = TinySCF->nthreads;
	
	double t0, t1, t2;
	
	#pragma omp parallel
	{
		int tid  = omp_get_thread_num();

		#pragma omp master
		t0 = get_wtime_sec();
		
		// Use thread local buffer (aligned to 128B) to reduce false sharing
		double *temp_J_thread = TinySCF->temp_J + TinySCF->df_nbf_16 * tid;
		
		// Generate temporary array for J
		memset(temp_J_thread, 0, sizeof(double) * df_nbf);

		#pragma omp for
		for (int kl = 0; kl < nbf * nbf; kl++)
		{
			int l = kl % nbf;
			int k = kl / nbf;
			
			double D_kl = D_mat[k * nbf + l];
			size_t offset = (size_t) (l * nbf + k) * (size_t) df_nbf;
			double *df_tensor_row = df_tensor + offset;

			#pragma simd
			for (size_t p = 0; p < df_nbf; p++)
				temp_J_thread[p] += D_kl * df_tensor_row[p];
		}
		
		#pragma omp barrier
		reduce_temp_J(TinySCF->temp_J, temp_J_thread, TinySCF->df_nbf_16, tid, TinySCF->nthreads);
		
		#pragma omp master
		t1 = get_wtime_sec();

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
		
		#pragma omp master
		t2 = get_wtime_sec();
	}
	
	*temp_J_t = t1 - t0;
	*J_mat_t  = t2 - t1;
}


// Generate the temporary tensor for K matrix and form K matrix using D matrix
// This procedure should only be called in the 1st to use the initial guess
// High flop-per-byte ratio: access: nbf^2 * (2*df_nbf+1), compute: nbf^3 * df_nbf
// Note: the K_mat is not completed, the symmetrizing is done later
static void build_K_mat_D(TinySCF_t TinySCF, double *temp_K_t, double *K_mat_t)
{
	double *K_mat     = TinySCF->K_mat;
	double *D_mat     = TinySCF->D_mat;
	double *df_tensor = TinySCF->df_tensor;
	double *temp_K    = TinySCF->temp_K;
	int nbf           = TinySCF->nbasfuncs;
	int df_nbf        = TinySCF->df_nbf;
	
	double t0, t1, t2;
	
	t0 = get_wtime_sec();
	
	// Construct temporary tensor for K matrix
	// Formula: temp_K(k, j, p) = dot(D_mat(k, 1:nbf), df_tensor(1:nbf, j, p))
	const int group_size = nbf;
	const CBLAS_TRANSPOSE temp_K_transa = CblasNoTrans;
	const CBLAS_TRANSPOSE temp_K_transb = CblasNoTrans;
	const int temp_K_m = nbf, temp_K_n = df_nbf, temp_K_k = nbf;
	const double temp_K_alpha = 1.0, temp_K_beta = 0.0;
	const int temp_K_lda = nbf;
	const int temp_K_ldb = nbf * df_nbf;
	const int temp_K_ldc = nbf * df_nbf;
	cblas_dgemm_batch(
		CblasRowMajor, &temp_K_transa, &temp_K_transb, 
		&temp_K_m, &temp_K_n, &temp_K_k, 
		&temp_K_alpha, 
		(const double **) TinySCF->temp_K_a, &temp_K_lda,
		(const double **) TinySCF->temp_K_b, &temp_K_ldb,
		&temp_K_beta,
		TinySCF->temp_K_c, &temp_K_ldc,
		1, &group_size
	);

	t1 = get_wtime_sec();
	
	// Build K matrix
	// Formula: K(i, j) = sum_{k=1}^{nbf} [ dot(df_tensor(i, k, 1:df_nbf), temp_K(k, j, 1:df_nbf)) ]
	memset(K_mat, 0, DBL_SIZE * nbf * nbf);
	
	for (int k = 0; k < nbf; k++)
	{
		cblas_dgemm_batch(
			CblasRowMajor, TinySCF->mat_K_transa, TinySCF->mat_K_transb, 
			TinySCF->mat_K_m, TinySCF->mat_K_n, TinySCF->mat_K_k, 
			TinySCF->mat_K_alpha, 
			(const double **) TinySCF->mat_K_a, TinySCF->mat_K_lda,
			(const double **) TinySCF->mat_K_b, TinySCF->mat_K_ldb,
			TinySCF->mat_K_beta,
			TinySCF->mat_K_c, TinySCF->mat_K_ldc,
			3, TinySCF->mat_K_group_size
		);
		
		size_t K_a_offset = df_nbf;
		size_t K_b_offset = nbf * df_nbf;
		for (int i = 0; i < TinySCF->mat_K_ntiles; i++)
		{
			TinySCF->mat_K_a[i] += K_a_offset;
			TinySCF->mat_K_b[i] += K_b_offset;
		}
	}
	// Reset array points to initial values
	size_t K_a_offset = (size_t) nbf * df_nbf;
	size_t K_b_offset = (size_t) nbf * (size_t) nbf * (size_t) df_nbf;
	for (int i = 0; i < TinySCF->mat_K_ntiles; i++)
	{
		TinySCF->mat_K_a[i] -= K_a_offset;
		TinySCF->mat_K_b[i] -= K_b_offset;
	}
	
	t2 = get_wtime_sec();
	
	*temp_K_t = t1 - t0;
	*K_mat_t  = t2 - t1;
}

// Generate the temporary tensor for K matrix and form K matrix using D matrix
// High flop-per-byte ratio: access: nbf * df_nbf * (nbf + n_occ) , compute: nbf^2 * df_nbf * n_occ
// Note: the K_mat is not completed, the symmetrizing is done later
static void build_K_mat(TinySCF_t TinySCF, double *temp_K_t, double *K_mat_t)
{
	double *K_mat = TinySCF->K_mat;
	int nbf       = TinySCF->nbasfuncs;
	int df_nbf    = TinySCF->df_nbf;
	int n_occ     = TinySCF->n_occ;
	
	double t0, t1, t2;
	
	t0 = get_wtime_sec();
	
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
		(const double **) TinySCF->temp_K_a, &temp_K_lda,
		(const double **) TinySCF->temp_K_b, &temp_K_ldb,
		&temp_K_beta,
		TinySCF->temp_K_c, &temp_K_ldc,
		1, &group_size
	);
	
	t1 = get_wtime_sec();

	// Build K matrix
	// Formula: K(i, j) = sum_{s=1}^{n_occ} [ dot(temp_K(s, i, 1:df_nbf), temp_K(s, j, 1:df_nbf)) ]
	memset(K_mat, 0, DBL_SIZE * nbf * nbf);
	for (int s = 0; s < n_occ; s++)
	{
		cblas_dgemm_batch(
			CblasRowMajor, TinySCF->mat_K_transa, TinySCF->mat_K_transb, 
			TinySCF->mat_K_m, TinySCF->mat_K_n, TinySCF->mat_K_k, 
			TinySCF->mat_K_alpha, 
			(const double **) TinySCF->mat_K_a, TinySCF->mat_K_lda,
			(const double **) TinySCF->mat_K_b, TinySCF->mat_K_ldb,
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
	
	t2 = get_wtime_sec();
	
	*temp_K_t = t1 - t0;
	*K_mat_t  = t2 - t1;
}

static void reset_batch_dgemm_arrays(TinySCF_t TinySCF)
{
	int nbf    = TinySCF->nbasfuncs;
	int df_nbf = TinySCF->df_nbf;
	int n_occ  = TinySCF->n_occ;
	
	for (int i = 0; i < nbf; i++)
	{
		size_t offset_b = (size_t) i * (size_t) nbf * (size_t) df_nbf;
		size_t offset_c = (size_t) i * (size_t) df_nbf;
		TinySCF->temp_K_a[i] = TinySCF->Cocc_mat;
		TinySCF->temp_K_b[i] = TinySCF->df_tensor + offset_b;
		TinySCF->temp_K_c[i] = TinySCF->temp_K    + offset_c;
	}
	
	int *group_size = TinySCF->mat_K_group_size;
	int mat_K_BS    = TinySCF->mat_K_BS;
	int cnt0 = 0, cnt1 = group_size[0];
	int cnt2 = group_size[0] + group_size[1];
	for (int i = 0; i < nbf; i += mat_K_BS)
	{
		int i_len = mat_K_BS < (nbf - i) ? mat_K_BS : (nbf - i);
		for (int j = i; j < nbf; j += mat_K_BS)
		{
			int j_len = mat_K_BS < (nbf - j) ? mat_K_BS : (nbf - j);
			
			// Use k = 0 as initial pointer position
			size_t offset_i0 = (size_t) i * (size_t) df_nbf;
			size_t offset_j0 = (size_t) j * (size_t) df_nbf;
			double *K_ij     = TinySCF->K_mat  + i * nbf + j;
			double *temp_K_i = TinySCF->temp_K + offset_i0;
			double *temp_K_j = TinySCF->temp_K + offset_j0;
			
			int cnt, gid;
			if ((i_len == mat_K_BS) && (j_len == mat_K_BS))
			{
				cnt = cnt0;
				gid = 0;
				cnt0++;
			} else {
				if ((i_len == mat_K_BS) && (j_len < mat_K_BS)) 
				{
					cnt = cnt1;
					gid = 1;
					cnt1++;
				} else {
					cnt = cnt2;
					gid = 2;
				}
			}
			
			TinySCF->mat_K_transa[gid] = CblasNoTrans;
			TinySCF->mat_K_transb[gid] = CblasTrans;
			TinySCF->mat_K_m[gid]      = i_len;
			TinySCF->mat_K_n[gid]      = j_len;
			TinySCF->mat_K_k[gid]      = df_nbf;
			TinySCF->mat_K_alpha[gid]  = 1.0;
			TinySCF->mat_K_beta[gid]   = 1.0;
			TinySCF->mat_K_a[cnt]      = temp_K_i;
			TinySCF->mat_K_b[cnt]      = temp_K_j;
			TinySCF->mat_K_c[cnt]      = K_ij;
			TinySCF->mat_K_lda[gid]    = df_nbf;
			TinySCF->mat_K_ldb[gid]    = df_nbf;
			TinySCF->mat_K_ldc[gid]    = nbf;
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
	
	double st0, et0, st1, build_F_t, temp_J_t, J_mat_t, temp_K_t, K_mat_t, symm_t;
	
	st0 = get_wtime_sec();
	
	// Build J matrix
	build_J_mat(TinySCF, &temp_J_t, &J_mat_t);
	
	// Build K matrix
	if (TinySCF->iter == 0)  // Use the initial guess D for 1st iteration
	{
		build_K_mat_D(TinySCF, &temp_K_t, &K_mat_t);
		reset_batch_dgemm_arrays(TinySCF);
	} else {
		build_K_mat(TinySCF, &temp_K_t, &K_mat_t);
	}
	
	// Symmetrizing J and K matrix and build Fock matrix
	st1 = get_wtime_sec();
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
	build_F_t = et0 - st0;
	symm_t    = et0 - st1;
	
	printf("* Build Fock matrix     : %.3lf (s)\n", build_F_t);
	printf("|---- aux. J tensor     |---- %.3lf (s)\n", temp_J_t);
	printf("|---- J matrix          |---- %.3lf (s)\n", J_mat_t);
	printf("|---- aux. K tensor     |---- %.3lf (s)\n", temp_K_t);
	printf("|---- K matrix          |---- %.3lf (s)\n", K_mat_t);
	printf("|---- J K symmetrizing  |---- %.3lf (s)\n", symm_t);
}