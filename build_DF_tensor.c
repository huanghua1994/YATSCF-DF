#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_DF_tensor.h"

void TinySCF_build_DF_tensor(TinySCF_t TinySCF)
{
	double *pqA = TinySCF->pqA;
	double *Jpq = TinySCF->Jpq;
	double *df_tensor = TinySCF->df_tensor;
	int nbf    = TinySCF->nbasfuncs;
	int df_nbf = TinySCF->df_nbf;
	int nshell    = TinySCF->nshells;
	int df_nshell = TinySCF->df_nshells;
	int *shell_bf_sind    = TinySCF->shell_bf_sind;
	int *df_shell_bf_sind = TinySCF->df_shell_bf_sind;
	Simint_t simint = TinySCF->simint;
	
	// Calculate 3-center density fitting integrals
	for (int M = 0; M < nshell; M++)
	{
		for (int N = M; N < nshell; N++)
		{
			int startM = shell_bf_sind[M];
			int endM   = shell_bf_sind[M + 1];
			int startN = shell_bf_sind[N];
			int endN   = shell_bf_sind[N + 1];
			int dimM   = endM - startM;
			int dimN   = endN - startN;
			for (int P = 0; P < df_nshell; P++)
			{
				double *integrals;
				int nints;
				int tid = 0;
				CMS_Simint_computeDFShellQuartet(
					simint, tid, M, N, P,
					&integrals, &nints
				);
				
				assert(nints > 0);
				
				int startP = df_shell_bf_sind[P];
				int dimP   = df_shell_bf_sind[P + 1] - startP;
				size_t row_mem_size = sizeof(double) * dimP;
				
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
				
			}  // for (int P = 0; P < df_nshell; P++)
		}  // for (int N = i; N < nshell; N++)
	}  // for (int M = 0; M < nshell; M++)
	
	// Calculate the Coulomb metric matrix
	double tmp1 = 0;
	for (int M = 0; M < df_nshell; M++)
	{
		for (int N = M; N < df_nshell; N++)
		{
			double *integrals;
			int nints;
			int tid = 0;
			CMS_Simint_computeDFShellPair(
				simint, tid, M, N, 
				&integrals, &nints
			);
			
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
}