#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <libgen.h>
#include <float.h>

#include <mkl.h>

#include "CMS.h"
#include "utils.h"
#include "TinySCF.h"

void init_TinySCF(TinySCF_t TinySCF, char *bas_fname, char *df_bas_fname, char *xyz_fname, const int niters)
{
	assert(TinySCF != NULL);
	
	double st = get_wtime_sec();
	
	// Reset statistic info
	TinySCF->mem_size       = 0.0;
	TinySCF->init_time      = 0.0;
	TinySCF->S_Hcore_time   = 0.0;
	TinySCF->shell_scr_time = 0.0;
	
	// Load basis set and molecule from input and get chemical system info
	CMS_createBasisSet(&(TinySCF->basis));
	CMS_createBasisSet(&(TinySCF->df_basis));
	CMS_loadChemicalSystem(TinySCF->basis, bas_fname, xyz_fname);
	CMS_loadChemicalSystem(TinySCF->df_basis, df_bas_fname, xyz_fname);
	TinySCF->natoms     = CMS_getNumAtoms   (TinySCF->basis);
	TinySCF->nshells    = CMS_getNumShells  (TinySCF->basis);
	TinySCF->nbasfuncs  = CMS_getNumFuncs   (TinySCF->basis);
	TinySCF->n_occ      = CMS_getNumOccOrb  (TinySCF->basis);
	TinySCF->charge     = CMS_getTotalCharge(TinySCF->basis);
	TinySCF->electron   = CMS_getNneutral   (TinySCF->basis);
	TinySCF->df_nbf     = CMS_getNumFuncs   (TinySCF->df_basis);
	TinySCF->df_nshells = CMS_getNumShells  (TinySCF->df_basis);
	char *basis_name    = basename(bas_fname);
	char *df_basis_name = basename(df_bas_fname);
	char *molecule_name = basename(xyz_fname);
	printf("Job information:\n");
	printf("    basis set            = %s\n", basis_name);
	printf("    DF basis set         = %s\n", df_basis_name);
	printf("    molecule             = %s\n", molecule_name);
	printf("    # atoms              = %d\n", TinySCF->natoms);
	printf("    # shells             = %d\n", TinySCF->nshells);
	printf("    # basis functions    = %d\n", TinySCF->nbasfuncs);
	printf("    # DF shells          = %d\n", TinySCF->df_nshells);
	printf("    # DF basis functions = %d\n", TinySCF->df_nbf);
	printf("    # occupied orbits    = %d\n", TinySCF->n_occ);
	printf("    # charge             = %d\n", TinySCF->charge);
	printf("    # electrons          = %d\n", TinySCF->electron);
	
	// Initialize OpenMP parallel info and buffer
	int maxAM, max_buf_entry_size, total_buf_size;
	maxAM = CMS_getMaxMomentum(TinySCF->basis);
	TinySCF->max_dim = (maxAM + 1) * (maxAM + 2) / 2;
	max_buf_entry_size      = TinySCF->max_dim * TinySCF->max_dim;
	TinySCF->nthreads       = omp_get_max_threads();
	TinySCF->max_buf_size   = max_buf_entry_size * 6;
	total_buf_size          = TinySCF->max_buf_size * TinySCF->nthreads;
	TinySCF->Accum_Fock_buf = ALIGN64B_MALLOC(DBL_SIZE * total_buf_size);
	assert(TinySCF->Accum_Fock_buf);
	TinySCF->mem_size += (double) TinySCF->max_buf_size;
	
	// Compute auxiliary variables
	TinySCF->nshellpairs = TinySCF->nshells   * TinySCF->nshells;
	TinySCF->mat_size    = TinySCF->nbasfuncs * TinySCF->nbasfuncs;
	TinySCF->num_uniq_sp = (TinySCF->nshells + 1) * TinySCF->nshells / 2;
	
	// Set SCF iteration info
	TinySCF->iter    = 0;
	TinySCF->niters  = niters;
	TinySCF->ene_tol = 1e-11;
	
	// Set screening thresholds, allocate memory for shell quartet screening 
	TinySCF->shell_scrtol2 = 1e-11 * 1e-11;
	TinySCF->sp_scrval     = (double*) ALIGN64B_MALLOC(DBL_SIZE * TinySCF->nshellpairs);
	TinySCF->df_sp_scrval  = (double*) ALIGN64B_MALLOC(DBL_SIZE * TinySCF->df_nshells);
	TinySCF->uniq_sp_lid   = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->num_uniq_sp);
	TinySCF->uniq_sp_rid   = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->num_uniq_sp);
	assert(TinySCF->sp_scrval     != NULL);
	assert(TinySCF->df_sp_scrval  != NULL);
	assert(TinySCF->uniq_sp_lid   != NULL);
	assert(TinySCF->uniq_sp_rid   != NULL);
	TinySCF->mem_size += (double) (DBL_SIZE * TinySCF->nshellpairs);
	TinySCF->mem_size += (double) (INT_SIZE * 2 * TinySCF->num_uniq_sp);
	
	// Initialize Simint object and shell basis function index info
	CMS_createSimint(TinySCF->basis, TinySCF->df_basis, &(TinySCF->simint), TinySCF->nthreads);
	TinySCF->shell_bf_sind = (int*) ALIGN64B_MALLOC(INT_SIZE * (TinySCF->nshells + 1));
	TinySCF->shell_bf_num  = (int*) ALIGN64B_MALLOC(INT_SIZE * TinySCF->nshells);
	assert(TinySCF->shell_bf_sind != NULL);
	assert(TinySCF->shell_bf_num  != NULL);
	TinySCF->mem_size += (double) (INT_SIZE * (2 * TinySCF->nshells + 1));
	for (int i = 0; i < TinySCF->nshells; i++)
	{
		TinySCF->shell_bf_sind[i] = CMS_getFuncStartInd(TinySCF->basis, i);
		TinySCF->shell_bf_num[i]  = CMS_getShellDim    (TinySCF->basis, i);
	}
	TinySCF->shell_bf_sind[TinySCF->nshells] = TinySCF->nbasfuncs;
	
	// Initialize density fitting shell basis function index info
	TinySCF->df_shell_bf_sind = (int*) ALIGN64B_MALLOC(INT_SIZE * (TinySCF->df_nshells + 1));
	TinySCF->df_shell_bf_num  = (int*) ALIGN64B_MALLOC(INT_SIZE * TinySCF->df_nshells);
	assert(TinySCF->df_shell_bf_sind != NULL);
	assert(TinySCF->df_shell_bf_num  != NULL);
	TinySCF->mem_size += (double) (INT_SIZE * (2 * TinySCF->df_nshells + 1));
	for (int i = 0; i < TinySCF->df_nshells; i++)
	{
		TinySCF->df_shell_bf_sind[i] = CMS_getFuncStartInd(TinySCF->df_basis, i);
		TinySCF->df_shell_bf_num[i]  = CMS_getShellDim    (TinySCF->df_basis, i);
	}
	TinySCF->df_shell_bf_sind[TinySCF->df_nshells] = TinySCF->df_nbf;
	
	// Allocate memory for matrices and temporary arrays used in SCF
	size_t mat_mem_size    = DBL_SIZE * TinySCF->mat_size;
	TinySCF->Hcore_mat     = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->S_mat         = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->F_mat         = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->D_mat         = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->J_mat         = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->K_mat         = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->X_mat         = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->tmp_mat       = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->Cocc_mat      = (double*) ALIGN64B_MALLOC(DBL_SIZE * TinySCF->n_occ * TinySCF->nbasfuncs);
	TinySCF->eigval        = (double*) ALIGN64B_MALLOC(DBL_SIZE * TinySCF->nbasfuncs);
	TinySCF->ev_idx        = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nbasfuncs);
	assert(TinySCF->Hcore_mat     != NULL);
	assert(TinySCF->S_mat         != NULL);
	assert(TinySCF->F_mat         != NULL);
	assert(TinySCF->D_mat         != NULL);
	assert(TinySCF->J_mat         != NULL);
	assert(TinySCF->K_mat         != NULL);
	assert(TinySCF->X_mat         != NULL);
	assert(TinySCF->tmp_mat       != NULL);
	assert(TinySCF->Cocc_mat      != NULL);
	assert(TinySCF->eigval        != NULL);
	TinySCF->mem_size += (double) (8 * mat_mem_size);
	TinySCF->mem_size += (double) (DBL_SIZE * TinySCF->n_occ * TinySCF->nbasfuncs);
	TinySCF->mem_size += (double) ((DBL_SIZE + INT_SIZE) * TinySCF->nbasfuncs);
	
	// Allocate memory for matrices and temporary arrays used in DIIS
	size_t DIIS_row_memsize = DBL_SIZE * (MAX_DIIS + 1);
	TinySCF->F0_mat    = (double*) ALIGN64B_MALLOC(mat_mem_size * MAX_DIIS);
	TinySCF->R_mat     = (double*) ALIGN64B_MALLOC(mat_mem_size * MAX_DIIS);
	TinySCF->B_mat     = (double*) ALIGN64B_MALLOC(DIIS_row_memsize * (MAX_DIIS + 1));
	TinySCF->FDS_mat   = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->DIIS_rhs  = (double*) ALIGN64B_MALLOC(DIIS_row_memsize);
	TinySCF->DIIS_ipiv = (int*)    ALIGN64B_MALLOC(INT_SIZE * (MAX_DIIS + 1));
	assert(TinySCF->F0_mat    != NULL);
	assert(TinySCF->R_mat     != NULL);
	assert(TinySCF->B_mat     != NULL);
	assert(TinySCF->DIIS_rhs  != NULL);
	assert(TinySCF->DIIS_ipiv != NULL);
	TinySCF->mem_size += MAX_DIIS * 2 * (double) mat_mem_size;
	TinySCF->mem_size += (double) DIIS_row_memsize * (MAX_DIIS + 2);
	TinySCF->mem_size += (double) INT_SIZE * (MAX_DIIS + 1);
	TinySCF->mem_size += (double) mat_mem_size;
	// Must initialize F0 and R as 0 
	memset(TinySCF->F0_mat, 0, mat_mem_size * MAX_DIIS);
	memset(TinySCF->R_mat,  0, mat_mem_size * MAX_DIIS);
	TinySCF->DIIS_len = 0;
	// Initialize B_mat
	for (int i = 0; i < (MAX_DIIS + 1) * (MAX_DIIS + 1); i++)
		TinySCF->B_mat[i] = -1.0;
	for (int i = 0; i < (MAX_DIIS + 1); i++)
		TinySCF->B_mat[i * (MAX_DIIS + 1) + i] = 0.0;
	TinySCF->DIIS_bmax_id = 0;
	TinySCF->DIIS_bmax    = -DBL_MAX;
	
	// Allocate memory for density fitting tensors and buffers
	TinySCF->df_nbf_16 = (TinySCF->df_nbf + 15) / 16 * 16;
	size_t temp_J0_memsize = (size_t) TinySCF->df_nbf_16 * (size_t) TinySCF->nthreads;
	size_t tensor_memsize  = (size_t) TinySCF->mat_size  * (size_t) TinySCF->df_nbf;
	size_t df_mat_memsize  = (size_t) TinySCF->df_nbf    * (size_t) TinySCF->df_nbf;
	tensor_memsize  *= DBL_SIZE;
	df_mat_memsize  *= DBL_SIZE;
	temp_J0_memsize *= DBL_SIZE;
	size_t Jpq_J0_memsize = temp_J0_memsize > df_mat_memsize ? temp_J0_memsize : df_mat_memsize;
	TinySCF->pqA       = (double*) ALIGN64B_MALLOC(tensor_memsize);
	TinySCF->df_tensor = (double*) ALIGN64B_MALLOC(tensor_memsize);
	TinySCF->Jpq       = (double*) ALIGN64B_MALLOC(Jpq_J0_memsize);
	assert(TinySCF->pqA       != NULL);
	assert(TinySCF->df_tensor != NULL);
	assert(TinySCF->Jpq       != NULL);
	TinySCF->mem_size += (double) tensor_memsize * 2;
	TinySCF->mem_size += (double) df_mat_memsize;
	// Jpq and pqA is no longer needed after df_tensor is generated,
	// use them as the buffer for Fock build
	TinySCF->temp_J = TinySCF->Jpq;
	TinySCF->temp_K = TinySCF->pqA;
	
	double et = get_wtime_sec();
	TinySCF->init_time = et - st;
	
	// Print memory usage and time consumption
	printf("TinySCF memory usage    = %.2lf MB\n", TinySCF->mem_size / 1048576.0);
	printf("TinySCF memory allocation and initialization over, elapsed time = %.3lf (s)\n", TinySCF->init_time);
}


void free_TinySCF(TinySCF_t TinySCF)
{
	assert(TinySCF != NULL);
	
	// Free Fock accumulation buffer
	ALIGN64B_FREE(TinySCF->Accum_Fock_buf);
	
	// Free shell quartet screening arrays
	ALIGN64B_FREE(TinySCF->sp_scrval);
	ALIGN64B_FREE(TinySCF->uniq_sp_lid);
	ALIGN64B_FREE(TinySCF->uniq_sp_rid);
	
	// Free shell basis function index info arrays
	ALIGN64B_FREE(TinySCF->shell_bf_sind);
	ALIGN64B_FREE(TinySCF->shell_bf_num);
	ALIGN64B_FREE(TinySCF->df_shell_bf_sind);
	ALIGN64B_FREE(TinySCF->df_shell_bf_num);
	
	// Free matrices and temporary arrays used in SCF
	ALIGN64B_FREE(TinySCF->Hcore_mat);
	ALIGN64B_FREE(TinySCF->S_mat);
	ALIGN64B_FREE(TinySCF->F_mat);
	ALIGN64B_FREE(TinySCF->D_mat);
	ALIGN64B_FREE(TinySCF->J_mat);
	ALIGN64B_FREE(TinySCF->K_mat);
	ALIGN64B_FREE(TinySCF->X_mat);
	ALIGN64B_FREE(TinySCF->tmp_mat);
	ALIGN64B_FREE(TinySCF->Cocc_mat);
	ALIGN64B_FREE(TinySCF->eigval);
	ALIGN64B_FREE(TinySCF->ev_idx);
	
	// Free matrices and temporary arrays used in DIIS
	ALIGN64B_FREE(TinySCF->F0_mat);
	ALIGN64B_FREE(TinySCF->R_mat);
	ALIGN64B_FREE(TinySCF->B_mat);
	ALIGN64B_FREE(TinySCF->FDS_mat);
	ALIGN64B_FREE(TinySCF->DIIS_rhs);
	ALIGN64B_FREE(TinySCF->DIIS_ipiv);

	// Free density fitting tensors and buffers
	ALIGN64B_FREE(TinySCF->pqA);
	ALIGN64B_FREE(TinySCF->Jpq);
	ALIGN64B_FREE(TinySCF->df_tensor);
	
	// Free BasisSet_t and Simint_t object, require Simint_t object print stat info
	CMS_destroyBasisSet(TinySCF->basis);
	CMS_destroyBasisSet(TinySCF->df_basis);
	CMS_destroySimint(TinySCF->simint, 1);
	
	free(TinySCF);
}



void TinySCF_init_batch_dgemm_arrays(TinySCF_t TinySCF)
{
	int nbf    = TinySCF->nbasfuncs;
	int df_nbf = TinySCF->df_nbf;
	
	// Batch dgemm arrays for temp_K construction
	TinySCF->temp_K_a = (double**) malloc(sizeof(double*) * nbf);
	TinySCF->temp_K_b = (double**) malloc(sizeof(double*) * nbf);
	TinySCF->temp_K_c = (double**) malloc(sizeof(double*) * nbf);
	assert(TinySCF->temp_K_a != NULL);
	assert(TinySCF->temp_K_b != NULL);
	assert(TinySCF->temp_K_c != NULL);
	// These values are only for 1st iteration
	for (int i = 0; i < nbf; i++)
	{
		TinySCF->temp_K_a[i] = TinySCF->D_mat;
		TinySCF->temp_K_b[i] = TinySCF->df_tensor + i * df_nbf;
		TinySCF->temp_K_c[i] = TinySCF->temp_K    + i * df_nbf;
	}
	
	// Batch dgemm arrays for mat_K construction
	int mat_K_BS = nbf / 16;
	if (mat_K_BS < 32) mat_K_BS = 32;
	int nblocks = (nbf + mat_K_BS - 1) / mat_K_BS;
	int last_block_size = nbf % mat_K_BS;
	int nblocks0 = nbf / mat_K_BS;
	int ntiles = (nblocks + 1) * nblocks / 2;
	int *group_size = TinySCF->mat_K_group_size;
	TinySCF->mat_K_ntiles = ntiles;
	TinySCF->mat_K_BS = mat_K_BS;
	group_size[0] = (nblocks0 * (nblocks0 + 1)) / 2;
	if (last_block_size > 0)
	{
		group_size[1] = nblocks0;
		group_size[2] = 1;
	} else {
		group_size[1] = 0;
		group_size[2] = 0;
	}
	TinySCF->mat_K_transa = (CBLAS_TRANSPOSE*) malloc(sizeof(CBLAS_TRANSPOSE) * 3);
	TinySCF->mat_K_transb = (CBLAS_TRANSPOSE*) malloc(sizeof(CBLAS_TRANSPOSE) * 3);
	TinySCF->mat_K_m      = (int*)     malloc(sizeof(int)     * 3);
	TinySCF->mat_K_n      = (int*)     malloc(sizeof(int)     * 3);
	TinySCF->mat_K_k      = (int*)     malloc(sizeof(int)     * 3);
	TinySCF->mat_K_alpha  = (double*)  malloc(sizeof(double)  * 3);
	TinySCF->mat_K_beta   = (double*)  malloc(sizeof(double)  * 3);
	TinySCF->mat_K_a      = (double**) malloc(sizeof(double*) * ntiles);
	TinySCF->mat_K_b      = (double**) malloc(sizeof(double*) * ntiles);
	TinySCF->mat_K_c      = (double**) malloc(sizeof(double*) * ntiles);
	TinySCF->mat_K_lda    = (int*)     malloc(sizeof(int)     * 3);
	TinySCF->mat_K_ldb    = (int*)     malloc(sizeof(int)     * 3);
	TinySCF->mat_K_ldc    = (int*)     malloc(sizeof(int)     * 3);
	assert(TinySCF->mat_K_transa != NULL);
	assert(TinySCF->mat_K_transb != NULL);
	assert(TinySCF->mat_K_m      != NULL);
	assert(TinySCF->mat_K_n      != NULL);
	assert(TinySCF->mat_K_k      != NULL);
	assert(TinySCF->mat_K_alpha  != NULL);
	assert(TinySCF->mat_K_beta   != NULL);
	assert(TinySCF->mat_K_a      != NULL);
	assert(TinySCF->mat_K_b      != NULL);
	assert(TinySCF->mat_K_c      != NULL);
	assert(TinySCF->mat_K_lda    != NULL);
	assert(TinySCF->mat_K_ldb    != NULL);
	assert(TinySCF->mat_K_ldc    != NULL);
	// These values are only for 1st iteration.
	// These arrays except a, b, c matrix pointers have fixed values, 
	// for matrix pointers, we can change them in the loop
	int cnt0 = 0, cnt1 = group_size[0];
	int cnt2 = group_size[0] + group_size[1];
	for (int i = 0; i < nbf; i += mat_K_BS)
	{
		int i_len = mat_K_BS < (nbf - i) ? mat_K_BS : (nbf - i);
		for (int j = i; j < nbf; j += mat_K_BS)
		{
			int j_len = mat_K_BS < (nbf - j) ? mat_K_BS : (nbf - j);
			
			// Use k = 0 as initial pointer position
			size_t offset_i0 = (size_t) (i * nbf + 0) * (size_t) df_nbf;
			size_t offset_0j = (size_t) (0 * nbf + j) * (size_t) df_nbf;
			double *K_ij        = TinySCF->K_mat     + i * nbf + j;
			double *df_tensor_i = TinySCF->df_tensor + offset_i0;
			double *temp_K_j    = TinySCF->temp_K    + offset_0j;
			
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
			TinySCF->mat_K_a[cnt]      = df_tensor_i;
			TinySCF->mat_K_b[cnt]      = temp_K_j;
			TinySCF->mat_K_c[cnt]      = K_ij;
			TinySCF->mat_K_lda[gid]    = nbf * df_nbf;
			TinySCF->mat_K_ldb[gid]    = df_nbf;
			TinySCF->mat_K_ldc[gid]    = nbf;
		}
	}
}

void TinySCF_free_batch_dgemm_arrays(TinySCF_t TinySCF)
{
	free(TinySCF->temp_K_a);
	free(TinySCF->temp_K_b);
	free(TinySCF->temp_K_c);
	
	free(TinySCF->mat_K_transa);
	free(TinySCF->mat_K_transb); 
	free(TinySCF->mat_K_m);
	free(TinySCF->mat_K_n);
	free(TinySCF->mat_K_k);
	free(TinySCF->mat_K_alpha);
	free(TinySCF->mat_K_beta);
	free(TinySCF->mat_K_a);
	free(TinySCF->mat_K_b);
	free(TinySCF->mat_K_c);
	free(TinySCF->mat_K_lda);
	free(TinySCF->mat_K_ldb);
	free(TinySCF->mat_K_ldc);
}

