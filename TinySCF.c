#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <libgen.h>
#include <float.h>

#include <mkl.h>

#include "CMS.h"
#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"
#include "build_DF_tensor.h"
#include "build_density.h"
#include "DIIS.h"

// This file only contains functions that initializing TinySCF engine, precomputing reusable 
// matrices and arrays and destroying TinySCF engine. Most time consuming functions are in
// build_Fock.c, build_density.c and DIIS.c

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
	
	// Allocate memory for blocked J, K and D matrices and the offsets of each block
	// and compute the offsets of each block of J, K and D matrices
	size_t MN_band_mem_size  = DBL_SIZE * TinySCF->max_dim * TinySCF->nbasfuncs;
	TinySCF->mat_block_ptr   = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nshellpairs);
	TinySCF->J_mat_block     = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->K_mat_block     = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->D_mat_block     = (double*) ALIGN64B_MALLOC(mat_mem_size);
	TinySCF->F_M_band_blocks = (double*) ALIGN64B_MALLOC(MN_band_mem_size * TinySCF->nthreads);
	TinySCF->F_N_band_blocks = (double*) ALIGN64B_MALLOC(MN_band_mem_size * TinySCF->nthreads);
	TinySCF->visited_Mpairs  = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nshells * TinySCF->nthreads);
	TinySCF->visited_Npairs  = (int*)    ALIGN64B_MALLOC(INT_SIZE * TinySCF->nshells * TinySCF->nthreads);
	assert(TinySCF->mat_block_ptr   != NULL);
	assert(TinySCF->J_mat_block     != NULL);
	assert(TinySCF->K_mat_block     != NULL);
	assert(TinySCF->D_mat_block     != NULL);
	assert(TinySCF->F_M_band_blocks != NULL);
	assert(TinySCF->F_N_band_blocks != NULL);
	assert(TinySCF->visited_Mpairs  != NULL);
	assert(TinySCF->visited_Npairs  != NULL);
	TinySCF->mem_size += (double) (3 * mat_mem_size);
	TinySCF->mem_size += (double) (INT_SIZE * TinySCF->nshellpairs);
	TinySCF->mem_size += (double) (2 * MN_band_mem_size * TinySCF->nthreads);
	TinySCF->mem_size += (double) (2 * INT_SIZE * TinySCF->nshells * TinySCF->nthreads);
	int pos = 0, idx = 0;
	for (int i = 0; i < TinySCF->nshells; i++)
	{
		for (int j = 0; j < TinySCF->nshells; j++)
		{
			TinySCF->mat_block_ptr[idx] = pos;
			pos += TinySCF->shell_bf_num[i] * TinySCF->shell_bf_num[j];
			idx++;
		}
	}
	
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
	size_t tensor_memsize = (size_t) TinySCF->mat_size * (size_t) TinySCF->df_nbf;
	size_t df_mat_memsize = (size_t) TinySCF->df_nbf * (size_t) TinySCF->df_nbf;
	tensor_memsize *= DBL_SIZE;
	df_mat_memsize *= DBL_SIZE;
	TinySCF->pqA       = (double*) ALIGN64B_MALLOC(tensor_memsize);
	TinySCF->df_tensor = (double*) ALIGN64B_MALLOC(tensor_memsize);
	TinySCF->Jpq       = (double*) ALIGN64B_MALLOC(df_mat_memsize);
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
	
	// Free blocked J, K and D matrices and the offsets of each block
	ALIGN64B_FREE(TinySCF->mat_block_ptr);
	ALIGN64B_FREE(TinySCF->J_mat_block);
	ALIGN64B_FREE(TinySCF->K_mat_block);
	ALIGN64B_FREE(TinySCF->D_mat_block);
	ALIGN64B_FREE(TinySCF->F_M_band_blocks);
	ALIGN64B_FREE(TinySCF->F_N_band_blocks);
	ALIGN64B_FREE(TinySCF->visited_Mpairs);
	ALIGN64B_FREE(TinySCF->visited_Npairs);
	
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

void TinySCF_compute_Hcore_Ovlp_mat(TinySCF_t TinySCF)
{
	assert(TinySCF != NULL);
	
	double st = get_wtime_sec();
	
	// Compute core Hamiltonian and overlap matrix
	memset(TinySCF->Hcore_mat, 0, DBL_SIZE * TinySCF->mat_size);
	memset(TinySCF->S_mat,     0, DBL_SIZE * TinySCF->mat_size);
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		#pragma omp for schedule(dynamic)
		for (int M = 0; M < TinySCF->nshells; M++)
		{
			for (int N = 0; N < TinySCF->nshells; N++)
			{
				int nints;
				double *integrals;
				
				int mat_topleft_offset = TinySCF->shell_bf_sind[M] * TinySCF->nbasfuncs + TinySCF->shell_bf_sind[N];
				double *S_mat_ptr      = TinySCF->S_mat     + mat_topleft_offset;
				double *Hcore_mat_ptr  = TinySCF->Hcore_mat + mat_topleft_offset;
				
				int nrows = TinySCF->shell_bf_num[M];
				int ncols = TinySCF->shell_bf_num[N];
				
				// Compute the contribution of current shell pair to core Hamiltonian matrix
				CMS_computePairOvl_Simint(TinySCF->basis, TinySCF->simint, tid, M, N, &integrals, &nints);
				if (nints > 0) copy_matrix_block(S_mat_ptr, TinySCF->nbasfuncs, integrals, ncols, nrows, ncols);
				
				// Compute the contribution of current shell pair to overlap matrix
				CMS_computePairCoreH_Simint(TinySCF->basis, TinySCF->simint, tid, M, N, &integrals, &nints);
				if (nints > 0) copy_matrix_block(Hcore_mat_ptr, TinySCF->nbasfuncs, integrals, ncols, nrows, ncols);
			}
		}
	}
	
	// Construct basis transformation 
	int N = TinySCF->nbasfuncs;
	double *U_mat  = TinySCF->tmp_mat; 
	double *U_mat0 = TinySCF->K_mat;    // K_mat is not used currently, use it as a temporary matrix
	double *eigval = TinySCF->eigval;
	// [U, D] = eig(S);
	memcpy(U_mat, TinySCF->S_mat, DBL_SIZE * TinySCF->mat_size);
	LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, U_mat, N, eigval); // U_mat will be overwritten by eigenvectors
	// X = U * D^{-1/2} * U'^T
	memcpy(U_mat0, U_mat, DBL_SIZE * TinySCF->mat_size);
	for (int i = 0; i < N; i++) 
		eigval[i] = 1.0 / sqrt(eigval[i]);
	for (int i = 0; i < N; i++)
	{
		#pragma simd
		for (int j = 0; j < N; j++)
			U_mat0[i * N + j] *= eigval[j];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, U_mat0, N, U_mat, N, 0.0, TinySCF->X_mat, N);
	
	double et = get_wtime_sec();
	TinySCF->S_Hcore_time = et - st;
	
	// Print runtime
	printf("TinySCF precompute Hcore, S, and X matrices over,  elapsed time = %.3lf (s)\n", TinySCF->S_Hcore_time);
}

static int cmp_pair(int M1, int N1, int M2, int N2)
{
	if (M1 == M2) return (N1 < N2);
	else return (M1 < M2);
}

static void quickSort(int *M, int *N, int l, int r)
{
	int i = l, j = r, tmp;
	int mid_M = M[(i + j) / 2];
	int mid_N = N[(i + j) / 2];
	while (i <= j)
	{
		while (cmp_pair(M[i], N[i], mid_M, mid_N)) i++;
		while (cmp_pair(mid_M, mid_N, M[j], N[j])) j--;
		if (i <= j)
		{
			tmp = M[i]; M[i] = M[j]; M[j] = tmp;
			tmp = N[i]; N[i] = N[j]; N[j] = tmp;
			
			i++;  j--;
		}
	}
	if (i < r) quickSort(M, N, i, r);
	if (j > l) quickSort(M, N, l, j);
}

void TinySCF_compute_sq_Schwarz_scrvals(TinySCF_t TinySCF)
{
	assert(TinySCF != NULL);
	
	double st = get_wtime_sec();
	
	// Compute screening values using Schwarz inequality
	double global_max_scrval = 0.0;
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		#pragma omp for schedule(dynamic) reduction(max:global_max_scrval)
		for (int M = 0; M < TinySCF->nshells; M++)
		{
			int dimM = TinySCF->shell_bf_num[M];
			for (int N = 0; N < TinySCF->nshells; N++)
			{
				int dimN = TinySCF->shell_bf_num[N];
				
				int nints;
				double *integrals;
				CMS_computeShellQuartet_Simint(TinySCF->simint, tid, M, N, M, N, &integrals, &nints);
				
				double maxval = 0.0;
				if (nints > 0)
				{
					// Loop over all ERIs in a shell quartet and find the max value
					for (int iM = 0; iM < dimM; iM++)
						for (int iN = 0; iN < dimN; iN++)
						{
							int index = iN * (dimM * dimN * dimM + dimM) + iM * (dimN * dimM + 1); // Simint layout
							double val = fabs(integrals[index]);
							if (val > maxval) maxval = val;
						}
				}
				TinySCF->sp_scrval[M * TinySCF->nshells + N] = maxval;
				if (maxval > global_max_scrval) global_max_scrval = maxval;
			}
		}
	}
	TinySCF->max_scrval = global_max_scrval;
	
	// Find the maximum screen value in density fitting shell pairs
	double global_max_df_scrval = 0;
	for (int i = 0; i < TinySCF->df_nshells; i++)
	{
		double df_scrval = CMS_Simint_getDFShellpairScreenVal(TinySCF->simint, i);
		TinySCF->df_sp_scrval[i] = df_scrval;
		if (df_scrval > global_max_df_scrval) 
			global_max_df_scrval = df_scrval;
	}
	
	// Reset Simint statistic info
	CMS_Simint_resetStatisInfo(TinySCF->simint);
	
	// Generate unique shell pairs that survive Schwarz screening
	// eta is the threshold for screening a shell pair
	double eta = TinySCF->shell_scrtol2 / global_max_df_scrval;
	int nnz = 0;
	for (int M = 0; M < TinySCF->nshells; M++)
	{
		for (int N = 0; N < TinySCF->nshells; N++)
		{
			double sp_scrval = TinySCF->sp_scrval[M * TinySCF->nshells + N];
			// if sp_scrval * max_scrval < shell_scrtol2, for any given shell pair
			// (P,Q), (MN|PQ) is always < shell_scrtol2 and will be screened
			if (sp_scrval > eta)  
			{
				// Make {N_i} in (M, N_i) as continuous as possible to get better
				// memory access pattern and better performance
				if (N < M) continue;
				
				// We want AM(M) >= AM(N) to avoid HRR
				int MN_id = CMS_Simint_getShellpairAMIndex(TinySCF->simint, M, N);
				int NM_id = CMS_Simint_getShellpairAMIndex(TinySCF->simint, N, M);
				if (MN_id > NM_id)
				{
					TinySCF->uniq_sp_lid[nnz] = M;
					TinySCF->uniq_sp_rid[nnz] = N;
				} else {
					TinySCF->uniq_sp_lid[nnz] = N;
					TinySCF->uniq_sp_rid[nnz] = M;
				}
				nnz++;
			}
		}
	}
	TinySCF->num_uniq_sp = nnz;
	quickSort(TinySCF->uniq_sp_lid, TinySCF->uniq_sp_rid, 0, nnz - 1);
	
	double et = get_wtime_sec();
	TinySCF->shell_scr_time = et - st;
	
	// Print runtime
	printf("TinySCF precompute shell screening info over,      elapsed time = %.3lf (s)\n", TinySCF->shell_scr_time);
}

void TinySCF_get_initial_guess(TinySCF_t TinySCF)
{
	memset(TinySCF->D_mat, 0, DBL_SIZE * TinySCF->mat_size);
	
	double *guess;
	int spos, epos, ldg;
	int nbf  = TinySCF->nbasfuncs;
	
	// Copy the SAD data to diagonal block of the density matrix
	for (int i = 0; i < TinySCF->natoms; i++)
	{
		CMS_getInitialGuess(TinySCF->basis, i, &guess, &spos, &epos);
		ldg = epos - spos + 1;
		double *D_mat_ptr = TinySCF->D_mat + spos * nbf + spos;
		copy_matrix_block(D_mat_ptr, nbf, guess, ldg, ldg, ldg);
	}
	
	// Scaling the initial density matrix according to the charge and neutral
	double R = 1.0;
	int charge   = TinySCF->charge;
	int electron = TinySCF->electron; 
	if (charge != 0 && electron != 0) 
		R = (double)(electron - charge) / (double)(electron);
	R *= 0.5;
	for (int i = 0; i < TinySCF->mat_size; i++)
		TinySCF->D_mat[i] *= R;
	
	// Calculate nuclear energy
	TinySCF->nuc_energy = CMS_getNucEnergy(TinySCF->basis);
}

// Compute Hartree-Fock energy
static void TinySCF_calc_energy(TinySCF_t TinySCF)
{
	double energy = 0.0;
	
	#pragma simd 
	for (int i = 0; i < TinySCF->mat_size; i++)
		energy += TinySCF->D_mat[i] * (TinySCF->F_mat[i] + TinySCF->Hcore_mat[i]);
	
	TinySCF->HF_energy = energy;
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
	// These arrays have fixed values, can be set at the beginning
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

void TinySCF_do_SCF(TinySCF_t TinySCF)
{
	TinySCF_build_DF_tensor(TinySCF);
	
	TinySCF_init_batch_dgemm_arrays(TinySCF);
	
	// Start SCF iterations
	printf("TinySCF SCF iteration started...\n");
	printf("Nuclear energy = %.10lf\n", TinySCF->nuc_energy);
	TinySCF->iter = 0;
	double prev_energy  = 0;
	double energy_delta = 223;
	while ((TinySCF->iter < TinySCF->niters) && (energy_delta >= TinySCF->ene_tol))
	{
		printf("--------------- Iteration %d ---------------\n", TinySCF->iter);
		
		double st0, et0, st1, et1;
		st0 = get_wtime_sec();
		
		// Build the Fock matrix
		st1 = get_wtime_sec();
		TinySCF_build_FockMat(TinySCF);
		et1 = get_wtime_sec();
		printf("* Build Fock matrix     : %.3lf (s)\n", et1 - st1);

		// Calculate new system energy
		st1 = get_wtime_sec();
		TinySCF_calc_energy(TinySCF);
		et1 = get_wtime_sec();
		printf("* Calculate energy      : %.3lf (s)\n", et1 - st1);
		energy_delta = fabs(TinySCF->HF_energy - prev_energy);
		prev_energy = TinySCF->HF_energy;

		// DIIS (Pulay mixing)
		st1 = get_wtime_sec();
		TinySCF_DIIS(TinySCF);
		et1 = get_wtime_sec();
		printf("* DIIS procedure        : %.3lf (s)\n", et1 - st1);
		
		// Diagonalize and build the density matrix
		st1 = get_wtime_sec();
		TinySCF_build_DenMat(TinySCF);
		et1 = get_wtime_sec();
		printf("* Build density matrix  : %.3lf (s)\n", et1 - st1);
		
		et0 = get_wtime_sec();
		
		printf("* Iteration runtime     = %.3lf (s)\n", et0 - st0);
		printf("* Energy = %.10lf (%.10lf)", TinySCF->HF_energy + TinySCF->nuc_energy, TinySCF->HF_energy);
		if (TinySCF->iter > 0) 
		{
			printf(", delta = %e\n", energy_delta); 
		} else 
		{
			printf("\n");
			energy_delta = 223;  // Prevent the SCF exit after 1st iteration when no SAD initial guess
		}
		
		TinySCF->iter++;
	}
	printf("--------------- SCF iterations finished ---------------\n");
	
	TinySCF_free_batch_dgemm_arrays(TinySCF);
}

