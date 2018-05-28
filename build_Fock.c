#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"

void TinySCF_build_FockMat(TinySCF_t TinySCF)
{
	double *Hcore_mat = TinySCF->Hcore_mat;
	double *J_mat     = TinySCF->J_mat;
	double *K_mat     = TinySCF->K_mat;
	double *D_mat     = TinySCF->D_mat;
	double *F_mat     = TinySCF->F_mat;
	double *df_tensor = TinySCF->df_tensor;
	double *temp_J    = TinySCF->temp_J;
	double *temp_K    = TinySCF->temp_K;
	int nbf           = TinySCF->nbasfuncs;
	int df_nbf        = TinySCF->df_nbf;

	memset(temp_J, 0, DBL_SIZE * df_nbf);
	memset(temp_K, 0, DBL_SIZE * nbf * nbf * df_nbf);

	// UNDONE: (1) parallelize; (2) use dgemm for K matrix contraction

	// Generate the temporary tensor for J and K matrix
	for (int k = 0; k < nbf; k++)
	{
		for (int l = 0; l < nbf; l++)
		{
			double D_kl = D_mat[k * nbf + l];
			double *df_tensor_row = df_tensor + (l * nbf + k) * df_nbf;
			#pragma simd
			for (int p = 0; p < df_nbf; p++)
				temp_J[p] += D_kl * df_tensor_row[p];
		}
	}

	for (int k = 0; k < nbf; k++)
	{
		for (int j = 0; j < nbf; j++)
		{
			for (int l = 0; l < nbf; l++)
			{
				double D_lk = D_mat[k * nbf + l];
				double *temp_K_row = temp_K + (k * nbf + j) * df_nbf;
				double *df_tensor_row = df_tensor + (l * nbf + j) * df_nbf;
				#pragma simd
				for (int p = 0; p < df_nbf; p++)
					temp_K_row[p] += D_lk * df_tensor_row[p];
			}
		}
	}

	// Build F and K matrix
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
			J_mat[j * nbf + i] = t;
		}
	}

	for (int i = 0; i < nbf; i++)
	{
		for (int j = i; j < nbf; j++)
		{
			double t = 0;
			
			for (int k = 0; k < nbf; k++)
			{
				double *temp_K_row = temp_K + (k * nbf + j) * df_nbf;
				double *df_tensor_row = df_tensor + (i * nbf + k) * df_nbf;
				#pragma simd
				for (int p = 0; p < df_nbf; p++)
					t += temp_K_row[p] * df_tensor_row[p];
			}

			K_mat[i * nbf + j] = t;
			K_mat[j * nbf + i] = t;
		}
	}

	// Build Fock matrix
	for (int i = 0; i < nbf * nbf; i++)
		F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] - K_mat[i];
}