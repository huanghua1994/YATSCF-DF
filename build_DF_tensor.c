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
				double *pqA_ptr0 = pqA + (iM * nbf + iN) * df_nbf + startP;
				double *pqA_ptr1 = pqA + (iN * nbf + iM) * df_nbf + startP;
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
						
						assert(thread_nints > 0);

						copy_3center_integral_results(
							thread_npairs, thread_P_list, thread_nints, thread_integrals,
							df_shell_bf_sind, pqA, nbf, df_nbf,
							startM, endM, startN, endN, dimN
						);
						
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
					
					assert(thread_nints > 0);
					
					copy_3center_integral_results(
						thread_npairs, thread_P_list, thread_nints, thread_integrals,
						df_shell_bf_sind, pqA, nbf, df_nbf,
						startM, endM, startN, endN, dimN
					);
				} 
			}  // for (int iAM = 0; iAM <= simint->df_max_am; iAM++)
		}  // for (int iMN = 0; iMN < TinySCF->num_uniq_sp; iMN++)
		
		CMS_Simint_freeThreadMultishellpair(&thread_multi_shellpair);
	}  // #pragma omp parallel 
	
	free(P_lists);
}

void TinySCF_build_DF_tensor(TinySCF_t TinySCF)
{
	double *pqA           = TinySCF->pqA;
	double *Jpq           = TinySCF->Jpq;
	double *df_tensor     = TinySCF->df_tensor;
	int nbf               = TinySCF->nbasfuncs;
	int df_nbf            = TinySCF->df_nbf;
	int nshell            = TinySCF->nshells;
	int df_nshell         = TinySCF->df_nshells;
	int *shell_bf_sind    = TinySCF->shell_bf_sind;
	int *df_shell_bf_sind = TinySCF->df_shell_bf_sind;
	Simint_t simint       = TinySCF->simint;
	int *uniq_sp_lid      = TinySCF->uniq_sp_lid;
	int *uniq_sp_rid      = TinySCF->uniq_sp_rid;
	int num_uniq_sp       = TinySCF->num_uniq_sp;
	double *sp_scrval     = TinySCF->sp_scrval;
	double *df_sp_scrval  = TinySCF->df_sp_scrval;
	double scrtol2        = TinySCF->shell_scrtol2;
	
	double st, et;

	printf("---------- DF tensor construction ----------\n");

	// Calculate 3-center density fitting integrals
	st = get_wtime_sec();
	calc_DF_3center_integrals(TinySCF);
	et = get_wtime_sec();
	printf("* TinySCF 3-center integral : %.3lf (s)\n", et - st);
	
	// Calculate the Coulomb metric matrix
	// UNDONE: (1) parallelize; (2) batching
	st = get_wtime_sec();
	for (int M = 0; M < df_nshell; M++)
	{
		double scrval0 = df_sp_scrval[M];
		for (int N = M; N < df_nshell; N++)
		{
			double scrval1 = df_sp_scrval[N];
			if (scrval0 * scrval1 < scrtol2) continue;
			
			double *integrals;
			int nints;
			int tid = 0;
			CMS_Simint_computeDFShellPair(simint, tid, M, N, &integrals, &nints);
			
			// if (nints == 0) continue;  // Shell quartet is screened
			assert(nints > 0);
			
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
					double I = integrals[im * dimN + in];
					Jpq[iM * df_nbf + iN] = I;
					Jpq[iN * df_nbf + iM] = I;
				}
			}
		}  // for (int N = i; N < df_nshell; N++)
	}  // for (int M = 0; M < df_nshell; M++)
	et = get_wtime_sec();
	printf("* TinySCF 2-center integral : %.3lf (s)\n", et - st);

	// Factorize the Jpq
	st = get_wtime_sec();
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
	memcpy(tmp_mat1, tmp_mat0, df_mat_mem_size);
	for (int irow = 0; irow < df_nbf; irow++)
	{
		double *tmp_mat0_ptr = tmp_mat0 + irow * df_nbf;
		for (int icol = 0; icol < df_nbf; icol++)
			tmp_mat0_ptr[icol] *= df_eigval[icol];
	}
	// Get Jpq^{-1/2} = U * S^{-1/2} * U', Jpq^{-1/2} is stored in Jpq
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, df_nbf, df_nbf, df_nbf, 
				1.0, tmp_mat0, df_nbf, tmp_mat1, df_nbf, 0.0, Jpq, df_nbf);
	ALIGN64B_FREE(tmp_mat0);
	ALIGN64B_FREE(tmp_mat1);
	ALIGN64B_FREE(df_eigval);
	et = get_wtime_sec();
	printf("* TinySCF matrix inv-sqrt   : %.3lf (s)\n", et - st);

	// Form the density fitting tensor
	// UNDONE: (1) parallelize; (2) use dgemm
	st = get_wtime_sec();
	memset(df_tensor, 0, DBL_SIZE * nbf * nbf * df_nbf);
	for (int M = 0; M < nbf; M++)
	{
		for (int N = M; N < nbf; N++)
		{
			double *pqA_ptr = pqA + (M * nbf + N) * df_nbf;
			double *df_tensor_MN = df_tensor + (M * nbf + N) * df_nbf;
			double *df_tensor_NM = df_tensor + (N * nbf + M) * df_nbf;

			for (int irow = 0; irow < df_nbf; irow++)
			{
				double *Jpq_ptr = Jpq + irow * df_nbf;
				double pqA_k = pqA_ptr[irow];
				for (int icol = 0; icol < df_nbf; icol++)
					df_tensor_MN[icol] += pqA_k * Jpq_ptr[icol];
			}
			memcpy(df_tensor_NM, df_tensor_MN, DBL_SIZE * df_nbf);
		}
	}
	et = get_wtime_sec();
	printf("* TinySCF build DF tensor   : %.3lf (s)\n", et - st);

	printf("---------- DF tensor construction finished ----------\n");
}