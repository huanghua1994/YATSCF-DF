#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_DF_tensor.h"

static void copy_3center_integral_results(
	int thread_npairs, int *thread_P_list, int thread_nints, double *thread_integrals, 
	int *df_shell_bf_sind, double *pqA, int nbf, int df_nbf,
	int startM, int endM, int startN, int endN, int dimN
)
{
	for (int ipair = 0; ipair < thread_npairs; ipair++)
	{
		int P = thread_P_list[ipair];
		int startP = df_shell_bf_sind[P];
		int dimP   = df_shell_bf_sind[P + 1] - startP;
		size_t row_mem_size = sizeof(double) * dimP;
		double *integrals = thread_integrals + thread_nints * ipair;
		
		for (int iM = startM; iM < endM; iM++)
		{
			int im = iM - startM;
			for (int iN = startN; iN < endN; iN++)
			{
				int in = iN - startN;
				double *eri_ptr = integrals + (im * dimN + in) * dimP;
				size_t pqA_offset0 = (size_t) (iM * nbf + iN) * (size_t) df_nbf + (size_t) startP;
				size_t pqA_offset1 = (size_t) (iN * nbf + iM) * (size_t) df_nbf + (size_t) startP;
				double *pqA_ptr0 = pqA + pqA_offset0;
				double *pqA_ptr1 = pqA + pqA_offset1;
				memcpy(pqA_ptr0, eri_ptr, row_mem_size);
				memcpy(pqA_ptr1, eri_ptr, row_mem_size);
			}
		}
	}
}

static void calc_DF_3center_integrals(TinySCF_t TinySCF)
{
	double *pqA           = TinySCF->pqA;
	int nbf               = TinySCF->nbasfuncs;
	int df_nbf            = TinySCF->df_nbf;
	int nshell            = TinySCF->nshells;
	int *shell_bf_sind    = TinySCF->shell_bf_sind;
	int *df_shell_bf_sind = TinySCF->df_shell_bf_sind;
	Simint_t simint       = TinySCF->simint;
	int *uniq_sp_lid      = TinySCF->uniq_sp_lid;
	int *uniq_sp_rid      = TinySCF->uniq_sp_rid;
	int num_uniq_sp       = TinySCF->num_uniq_sp;
	double *sp_scrval     = TinySCF->sp_scrval;
	double *df_sp_scrval  = TinySCF->df_sp_scrval;
	double scrtol2        = TinySCF->shell_scrtol2;
	
	int *P_lists = (int*) malloc(sizeof(int) * _Simint_NSHELL_SIMD * TinySCF->nthreads);
	assert(P_lists != NULL);
	
	#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
		int *thread_P_list = P_lists + tid * _Simint_NSHELL_SIMD;
		double *thread_integrals;
		int thread_nints, thread_npairs;
		void *thread_multi_shellpair;
		CMS_Simint_createThreadMultishellpair(&thread_multi_shellpair);
		
		#pragma omp for schedule(dynamic)
		for (int iMN = 0; iMN < num_uniq_sp; iMN++)
		{
			int M = uniq_sp_lid[iMN];
			int N = uniq_sp_rid[iMN];
			int startM = shell_bf_sind[M];
			int endM   = shell_bf_sind[M + 1];
			int startN = shell_bf_sind[N];
			int endN   = shell_bf_sind[N + 1];
			int dimM   = endM - startM;
			int dimN   = endN - startN;
			double scrval0 = sp_scrval[M * nshell + N];
			
			for (int iAM = 0; iAM <= simint->df_max_am; iAM++)
			{
				thread_npairs = 0;
				int iP_start = simint->df_am_shell_spos[iAM];
				int iP_end   = simint->df_am_shell_spos[iAM + 1];
				for (int iP = iP_start; iP < iP_end; iP++)
				{
					int P = simint->df_am_shell_id[iP];
					double scrval1 = df_sp_scrval[P];
					if (scrval0 * scrval1 < scrtol2) continue;
					
					thread_P_list[thread_npairs] = P;
					thread_npairs++;
					
					if (thread_npairs == _Simint_NSHELL_SIMD)
					{
						CMS_Simint_computeDFShellQuartetBatch(
							simint, tid, M, N, thread_P_list, thread_npairs, 
							&thread_integrals, &thread_nints,
							&thread_multi_shellpair
						);
						
						if (thread_nints > 0)
						{
							copy_3center_integral_results(
								thread_npairs, thread_P_list, thread_nints, thread_integrals,
								df_shell_bf_sind, pqA, nbf, df_nbf,
								startM, endM, startN, endN, dimN
							);
						}
						
						thread_npairs = 0;
					}
				}  // for (int iP = iP_start; iP < iP_end; iP++)
				
				if (thread_npairs > 0)
				{
					CMS_Simint_computeDFShellQuartetBatch(
						simint, tid, M, N, thread_P_list, thread_npairs, 
						&thread_integrals, &thread_nints,
						&thread_multi_shellpair
					);
					
					if (thread_nints > 0)
					{
						copy_3center_integral_results(
							thread_npairs, thread_P_list, thread_nints, thread_integrals,
							df_shell_bf_sind, pqA, nbf, df_nbf,
							startM, endM, startN, endN, dimN
						);
					}
				} 
			}  // for (int iAM = 0; iAM <= simint->df_max_am; iAM++)
		}  // for (int iMN = 0; iMN < TinySCF->num_uniq_sp; iMN++)
		
		CMS_Simint_freeThreadMultishellpair(&thread_multi_shellpair);
	}  // #pragma omp parallel 
	
	free(P_lists);
}

static void calc_DF_2center_integrals(TinySCF_t TinySCF)
{
	// Fast enough, need not to batch shell quartets
	double *Jpq           = TinySCF->Jpq;
	int df_nbf            = TinySCF->df_nbf;
	int df_nshell         = TinySCF->df_nshells;
	int *df_shell_bf_sind = TinySCF->df_shell_bf_sind;
	Simint_t simint       = TinySCF->simint;
	double *df_sp_scrval  = TinySCF->df_sp_scrval;
	double scrtol2        = TinySCF->shell_scrtol2;
	
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int thread_nints;
		double *thread_integrals;
		
		#pragma omp for schedule(dynamic)
		for (int M = 0; M < df_nshell; M++)
		{
			double scrval0 = df_sp_scrval[M];
			for (int N = M; N < df_nshell; N++)
			{
				double scrval1 = df_sp_scrval[N];
				if (scrval0 * scrval1 < scrtol2) continue;

				CMS_Simint_computeDFShellPair(simint, tid, M, N, &thread_integrals, &thread_nints);
				
				if (thread_nints <= 0) continue;
				
				int startM = df_shell_bf_sind[M];
				int endM   = df_shell_bf_sind[M + 1];
				int startN = df_shell_bf_sind[N];
				int endN   = df_shell_bf_sind[N + 1];
				int dimM   = endM - startM;
				int dimN   = endN - startN;
				
				for (int iM = startM; iM < endM; iM++)
				{
					int im = iM - startM;
					for (int iN = startN; iN < endN; iN++)
					{
						int in = iN - startN;
						double I = thread_integrals[im * dimN + in];
						Jpq[iM * df_nbf + iN] = I;
						Jpq[iN * df_nbf + iM] = I;
					}
				}
			}  // for (int N = i; N < df_nshell; N++)
		}  // for (int M = 0; M < df_nshell; M++)
	}  // #pragma omp parallel
}

static void calc_inverse_sqrt_Jpq(TinySCF_t TinySCF)
{
	double *Jpq = TinySCF->Jpq;
	int df_nbf  = TinySCF->df_nbf;
	
	size_t df_mat_mem_size = DBL_SIZE * df_nbf * df_nbf;
	double *tmp_mat0  = ALIGN64B_MALLOC(df_mat_mem_size);
	double *tmp_mat1  = ALIGN64B_MALLOC(df_mat_mem_size);
	double *df_eigval = ALIGN64B_MALLOC(DBL_SIZE * df_nbf);
	assert(tmp_mat0 != NULL && tmp_mat1 != NULL);
	// Diagonalize Jpq = U * S * U^T, the eigenvectors are stored in tmp_mat0
	memcpy(tmp_mat0, Jpq, df_mat_mem_size);
	LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', df_nbf, tmp_mat0, df_nbf, df_eigval);
	// Apply inverse square root to eigen values to get the inverse squart root of Jpq
	for (int i = 0; i < df_nbf; i++)
		df_eigval[i] = 1.0 / sqrt(df_eigval[i]);
	// Right multiply the S^{-1/2} to U
	#pragma omp parallel for
	for (int irow = 0; irow < df_nbf; irow++)
	{
		double *tmp_mat0_ptr = tmp_mat0 + irow * df_nbf;
		double *tmp_mat1_ptr = tmp_mat1 + irow * df_nbf;
		memcpy(tmp_mat1_ptr, tmp_mat0_ptr, DBL_SIZE * df_nbf);
		for (int icol = 0; icol < df_nbf; icol++)
			tmp_mat0_ptr[icol] *= df_eigval[icol];
	}
	// Get Jpq^{-1/2} = U * S^{-1/2} * U', Jpq^{-1/2} is stored in Jpq
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, df_nbf, df_nbf, df_nbf, 
				1.0, tmp_mat0, df_nbf, tmp_mat1, df_nbf, 0.0, Jpq, df_nbf);
	ALIGN64B_FREE(tmp_mat0);
	ALIGN64B_FREE(tmp_mat1);
	ALIGN64B_FREE(df_eigval);
}

// Formula: df_tensor(i, j, k) = dot(pqA(i, j, 1:df_nbf), Jpq_invsqrt(1:df_nbf, k))
static void generate_df_tensor(TinySCF_t TinySCF)
{
	double *df_tensor = TinySCF->df_tensor;
	double *pqA  = TinySCF->pqA;
	double *Jpq  = TinySCF->Jpq;
	int nbf      = TinySCF->nbasfuncs;
	int df_nbf   = TinySCF->df_nbf;
	int mat_K_BS = TinySCF->mat_K_BS;

	/*
	#pragma omp parallel for
	for (size_t i = 0; i < nbf * nbf * df_nbf; i++) 
		df_tensor[i] = 0;
	
	// Cannot use batch dgemm here since each time the size of A matrix is different
	for (int M = 0; M < nbf; M++)
	{
		size_t offset = (size_t) (M * nbf + M) * (size_t) df_nbf;
		double *pqA_ptr      = pqA       + offset;
		double *df_tensor_MM = df_tensor + offset;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf - M, df_nbf, df_nbf,
		            1.0, pqA_ptr, df_nbf, Jpq, df_nbf, 0.0, df_tensor_MM, df_nbf);
	}
	*/
	
	double **dft_A = (double **) malloc(sizeof(double*) * mat_K_BS);
	double **dft_B = (double **) malloc(sizeof(double*) * mat_K_BS);
	double **dft_C = (double **) malloc(sizeof(double*) * mat_K_BS);
	assert(dft_A != NULL && dft_B != NULL && dft_C != NULL);

	for (int M = 0; M < nbf; M += mat_K_BS)
	{
		int nrows = mat_K_BS;
		if (M + nrows > nbf) nrows = nbf - M;
		for (int i = 0; i < nrows; i++)
		{
			size_t offset = (size_t) ((M + i) * nbf + M) * (size_t) df_nbf;
			dft_A[i] = pqA       + offset;
			dft_B[i] = Jpq;
			dft_C[i] = df_tensor + offset;
		}

		const CBLAS_TRANSPOSE dft_transa = CblasNoTrans;
		const CBLAS_TRANSPOSE dft_transb = CblasNoTrans;
		const int dft_m = nbf - M, dft_n = df_nbf, dft_k = df_nbf;
		const double dft_alpha = 1.0, dft_beta = 0.0;
		const int dft_lda = df_nbf;
		const int dft_ldb = df_nbf;
		const int dft_ldc = df_nbf;
		cblas_dgemm_batch(
			CblasRowMajor, &dft_transa, &dft_transb,
			&dft_m, &dft_n, &dft_k, 
			&dft_alpha,
			(const double **) dft_A, &dft_lda,
			(const double **) dft_B, &dft_ldb,
			&dft_beta,
			dft_C, &dft_ldc,
			1, &nrows
		);
	}

	free(dft_A);
	free(dft_B);
	free(dft_C);

	#pragma omp parallel for schedule(dynamic)
	for (int M = 0; M < nbf; M++)
	{
		for (int N = M; N < nbf; N++)
		{
			size_t MN_offset = (size_t) (M * nbf + N) * (size_t) df_nbf;
			size_t NM_offset = (size_t) (N * nbf + M) * (size_t) df_nbf;
			double *df_tensor_MN = df_tensor + MN_offset;
			double *df_tensor_NM = df_tensor + NM_offset;
			memcpy(df_tensor_NM, df_tensor_MN, DBL_SIZE * df_nbf);
		}
	}
}

void TinySCF_build_DF_tensor(TinySCF_t TinySCF)
{
	double st, et;

	printf("---------- DF tensor construction ----------\n");

	// Calculate 3-center density fitting integrals
	st = get_wtime_sec();
	calc_DF_3center_integrals(TinySCF);
	et = get_wtime_sec();
	printf("* 3-center integral : %.3lf (s)\n", et - st);
	
	// Calculate the Coulomb metric matrix
	st = get_wtime_sec();
	calc_DF_2center_integrals(TinySCF);
	et = get_wtime_sec();
	printf("* 2-center integral : %.3lf (s)\n", et - st);

	// Factorize the Jpq
	st = get_wtime_sec();
	calc_inverse_sqrt_Jpq(TinySCF);
	et = get_wtime_sec();
	printf("* matrix inv-sqrt   : %.3lf (s)\n", et - st);

	// Form the density fitting tensor
	st = get_wtime_sec();
	generate_df_tensor(TinySCF);
	et = get_wtime_sec();
	printf("* build DF tensor   : %.3lf (s)\n", et - st);

	printf("---------- DF tensor construction finished ----------\n");
}