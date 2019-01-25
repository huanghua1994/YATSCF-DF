#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <mkl.h>

#include "CMS.h"
#include "utils.h"
#include "TinySCF.h"
#include "build_Fock.h"
#include "build_density.h"
#include "DIIS.h"

// This file only contains functions that initializing TinySCF engine, precomputing reusable 
// matrices and arrays and destroying TinySCF engine. Most time consuming functions are in
// build_DF_tensor.c, build_Fock.c, build_density.c and DIIS.c

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

static void quickSort_MNpair(int *M, int *N, int l, int r)
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
    if (i < r) quickSort_MNpair(M, N, i, r);
    if (j > l) quickSort_MNpair(M, N, l, j);
}

void TinySCF_compute_sq_Schwarz_scrvals(TinySCF_t TinySCF)
{
    assert(TinySCF != NULL);
    
    double st = get_wtime_sec();
    
    int nbf = TinySCF->nbasfuncs;
    int nshells = TinySCF->nshells;
    int *shell_bf_num  = TinySCF->shell_bf_num;
    int *shell_bf_sind = TinySCF->shell_bf_sind;
    double *sp_scrval  = TinySCF->sp_scrval;
    double *bf_pair_scrval = TinySCF->bf_pair_scrval;
    
    // Compute screening values using Schwarz inequality
    double global_max_scrval = 0.0;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic) reduction(max:global_max_scrval)
        for (int M = 0; M < nshells; M++)
        {
            int dimM = shell_bf_num[M];
            int M_bf_idx = shell_bf_sind[M];
            for (int N = 0; N < nshells; N++)
            {
                int dimN = shell_bf_num[N];
                int N_bf_idx = shell_bf_sind[N];
                
                int nints;
                double *integrals;
                CMS_computeShellQuartet_Simint(TinySCF->simint, tid, M, N, M, N, &integrals, &nints);
                
                double maxval = 0.0;
                if (nints > 0)
                {
                    // Loop over all ERIs in a shell quartet and find the max value
                    for (int iM = 0; iM < dimM; iM++)
                    {
                        for (int iN = 0; iN < dimN; iN++)
                        {
                            int int_idx = iN * (dimM * dimN * dimM + dimM) + iM * (dimN * dimM + 1); // Simint layout
                            double val = fabs(integrals[int_idx]);
                            int bf_idx = (M_bf_idx + iM) * nbf + (N_bf_idx + iN);
                            bf_pair_scrval[bf_idx] = val;
                            if (val > maxval) maxval = val;
                        }
                    }
                }
                sp_scrval[M * nshells + N] = maxval;
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
        if (df_scrval > global_max_df_scrval) global_max_df_scrval = df_scrval;
    }
    TinySCF->max_df_scrval = global_max_df_scrval;
    
    // Reset Simint statistic info
    CMS_Simint_resetStatisInfo(TinySCF->simint);
    
    // Generate unique shell pairs that survive Schwarz screening
    // eta is the threshold for screening a shell pair
    double eta = TinySCF->shell_scrtol2 / TinySCF->max_df_scrval;
    int *uniq_sp_lid = TinySCF->uniq_sp_lid;
    int *uniq_sp_rid = TinySCF->uniq_sp_rid;
    int sp_nnz = 0;
    for (int M = 0; M < nshells; M++)
    {
        int dimM = shell_bf_num[M];
        int M_bf_idx = shell_bf_sind[M];
        for (int N = 0; N < nshells; N++)
        {
            int dimN = shell_bf_num[N];
            int N_bf_idx = shell_bf_sind[N];
            
            double sp_scrval_MN = sp_scrval[M * nshells + N];
            // if sp_scrval_MN * max_scrval < shell_scrtol2, for any given shell pair
            // (P,Q), (MN|PQ) is always < shell_scrtol2 and will be screened
            if (sp_scrval_MN > eta)  
            {
                // Make {N_i} in (M, N_i) as continuous as possible to get better
                // memory access pattern and better performance
                if (N < M) continue;
                
                // We want AM(M) >= AM(N) to avoid HRR
                int MN_id = CMS_Simint_getShellpairAMIndex(TinySCF->simint, M, N);
                int NM_id = CMS_Simint_getShellpairAMIndex(TinySCF->simint, N, M);
                if (MN_id > NM_id)
                {
                    uniq_sp_lid[sp_nnz] = M;
                    uniq_sp_rid[sp_nnz] = N;
                } else {
                    uniq_sp_lid[sp_nnz] = N;
                    uniq_sp_rid[sp_nnz] = M;
                }
                sp_nnz++;
            }
        }
    }
    TinySCF->num_uniq_sp = sp_nnz;
    quickSort_MNpair(TinySCF->uniq_sp_lid, TinySCF->uniq_sp_rid, 0, sp_nnz - 1);
    
    double et = get_wtime_sec();
    TinySCF->shell_scr_time = et - st;
    printf("TinySCF precompute shell screening info over,      elapsed time = %.3lf (s)\n", TinySCF->shell_scr_time);
}
    
void TinySCF_prepare_sparsity(TinySCF_t TinySCF)
{
    int nbf = TinySCF->nbasfuncs;
    double *bf_pair_scrval = TinySCF->bf_pair_scrval;
    double eta = TinySCF->shell_scrtol2 / TinySCF->max_df_scrval;
    
    double st = get_wtime_sec();
    
    int bf_pair_nnz = 0;
    int *bf_pair_mask = TinySCF->bf_pair_mask;
    int *bf_pair_j    = TinySCF->bf_pair_j;
    int *bf_pair_diag = TinySCF->bf_pair_diag;
    int *bf_mask_displs = TinySCF->bf_mask_displs;
    bf_mask_displs[0] = 0;
    for (int i = 0; i < nbf; i++)
    {
        int offset_i = i * nbf;
        for (int j = 0; j < nbf; j++)
        {
            if (bf_pair_scrval[offset_i + j] > eta)
            {
                bf_pair_mask[offset_i + j] = bf_pair_nnz;
                bf_pair_j[bf_pair_nnz] = j;
                bf_pair_nnz++;
            } else {
                bf_pair_mask[offset_i + j] = -1;
            }
        }
        bf_pair_diag[i] = bf_pair_mask[offset_i + i];  // (i, i) always survives screening
        bf_mask_displs[i + 1] = bf_pair_nnz;
    }
    
    double sp_sparsity = (double) TinySCF->num_uniq_sp / (double) TinySCF->nshellpairs;
    double bf_pair_sparsity = (double) bf_pair_nnz / (double) TinySCF->mat_size;
    
    double et = get_wtime_sec();
    double ut = et - st;
    printf("TinySCF handling shell pair sparsity over,         elapsed time = %.3lf (s)\n", ut);
    
    st = get_wtime_sec();
    size_t tensor_memsize = (size_t) bf_pair_nnz * (size_t) TinySCF->df_nbf * DBL_SIZE;
    TinySCF->pqA       = (double*) ALIGN64B_MALLOC(tensor_memsize);
    TinySCF->df_tensor = (double*) ALIGN64B_MALLOC(tensor_memsize);
    assert(TinySCF->pqA       != NULL);
    assert(TinySCF->df_tensor != NULL);
    TinySCF->mem_size += (double) tensor_memsize * 2;
    et = get_wtime_sec();
    ut = et - st;
    
    printf("TinySCF memory allocation and initialization over, elapsed time = %.3lf (s)\n", ut);
    printf("TinySCF regular + density fitting memory usage = %.2lf MB \n", TinySCF->mem_size / 1048576.0);
    printf("#### Sparsity of basis function pairs = %lf, %lf\n", sp_sparsity, bf_pair_sparsity);
}

void TinySCF_D2Cocc(TinySCF_t TinySCF)
{
    double *D_mat    = TinySCF->D_mat;
    double *Chol_mat = TinySCF->tmp_mat;
    double *Cocc_mat = TinySCF->Cocc_mat;
    int    nbf       = TinySCF->nbasfuncs;
    int    n_occ     = TinySCF->n_occ;
    
    int *piv = (int*) malloc(sizeof(int) * nbf);
    int rank;
    memcpy(Chol_mat, D_mat, DBL_SIZE * TinySCF->mat_size);
    
    // TODO: implement a partial Cholesky decomposition with pivoting
    LAPACKE_dpstrf(LAPACK_ROW_MAJOR, 'L', nbf, Chol_mat, nbf, piv, &rank, 1e-12);
    
    for (int i = 0; i < n_occ; i++)
    {
        double *Cocc_row = Cocc_mat + i * n_occ;
        double *Chol_row = Chol_mat + i * nbf;
        for (int j = 0; j < i; j++) Cocc_row[j] = Chol_row[j];
        for (int j = i; j < n_occ; j++) Cocc_row[j] = 0.0;
    }
    for (int i = n_occ; i < nbf; i++)
    {
        double *Cocc_row = Cocc_mat + i * n_occ;
        double *Chol_row = Chol_mat + i * nbf;
        memcpy(Cocc_row, Chol_row, DBL_SIZE * n_occ);
    }
    
    free(piv);
}

void TinySCF_get_initial_guess(TinySCF_t TinySCF)
{
    memset(TinySCF->D_mat, 0, DBL_SIZE * TinySCF->mat_size);
    
    double *guess;
    int spos, epos, ldg;
    int nbf = TinySCF->nbasfuncs;
    double *D_mat = TinySCF->D_mat;
    
    double t0 = get_wtime_sec();
    
    // Copy the SAD data to diagonal block of the density matrix
    for (int i = 0; i < TinySCF->natoms; i++)
    {
        CMS_getInitialGuess(TinySCF->basis, i, &guess, &spos, &epos);
        ldg = epos - spos + 1;
        double *D_mat_ptr = D_mat + spos * nbf + spos;
        copy_matrix_block(D_mat_ptr, nbf, guess, ldg, ldg, ldg);
    }
    
    // Scaling the initial density matrix according to the charge and neutral
    double R = 1.0;
    int charge   = TinySCF->charge;
    int electron = TinySCF->electron; 
    if (charge != 0 && electron != 0) 
        R = (double)(electron - charge) / (double)(electron);
    R *= 0.5;
    for (int i = 0; i < TinySCF->mat_size; i++) D_mat[i] *= R;
    
    // Calculate nuclear energy
    TinySCF->nuc_energy = CMS_getNucEnergy(TinySCF->basis);
    
    double t1 = get_wtime_sec();
    
    // Use a (partial) Cholesky decomposition to transform initial D to C_occ
    TinySCF_D2Cocc(TinySCF);
    
    double t2 = get_wtime_sec();
    
    printf("TinySCF SAD initial guess: D mat / Cocc mat used %lf, %lf (s)\n", t1 - t0, t2 - t1);
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

void TinySCF_do_SCF(TinySCF_t TinySCF)
{
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
        // st1 = get_wtime_sec();
        TinySCF_build_FockMat(TinySCF);
        // et1 = get_wtime_sec();
        // printf("* Build Fock matrix     : %.3lf (s)\n", et1 - st1);

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
}
