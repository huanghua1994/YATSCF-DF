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
    quickSort_MNpair(TinySCF->uniq_sp_lid, TinySCF->uniq_sp_rid, 0, nnz - 1);
    
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
    int nbf = TinySCF->nbasfuncs;
    double *D_mat = TinySCF->D_mat;
    
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
