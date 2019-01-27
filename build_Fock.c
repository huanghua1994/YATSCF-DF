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
#include "build_density.h"

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
    //double *df_tensor = TinySCF->df_tensor;
    double *temp_J    = TinySCF->temp_J;
    int nbf           = TinySCF->nbasfuncs;
    int df_nbf        = TinySCF->df_nbf;
    int nthreads      = TinySCF->nthreads;
    int *bf_pair_mask = TinySCF->bf_pair_mask;
    double *df_tensor = TinySCF->df_tensor;
    
    double t0, t1, t2;
    int *bf_pair_j = TinySCF->bf_pair_j;
    int *bf_pair_diag = TinySCF->bf_pair_diag;
    int *bf_mask_displs = TinySCF->bf_mask_displs;
    
    #pragma omp parallel
    {
        int tid  = omp_get_thread_num();

        #pragma omp master
        t0 = get_wtime_sec();
        
        // Use thread local buffer (aligned to 128B) to reduce false sharing
        double *temp_J_thread = TinySCF->temp_J + TinySCF->df_nbf_16 * tid;
        
        // Generate temporary array for J
        memset(temp_J_thread, 0, sizeof(double) * df_nbf);
        
        #pragma omp for schedule(dynamic)
        for (int k = 0; k < nbf; k++)
        {
            int diag_k_idx = bf_pair_diag[k];
            int idx_kk = k * nbf + k;
            
            // Basis function pair (i, i) always survives screening
            size_t offset = (size_t) diag_k_idx * (size_t) df_nbf;
            double *df_tensor_row = df_tensor + offset;
            double D_kl = D_mat[idx_kk];
            #pragma simd
            for (size_t p = 0; p < df_nbf; p++)
                temp_J_thread[p] += D_kl * df_tensor_row[p];
            
            
            int row_k_epos = bf_mask_displs[k + 1];
            for (int l_idx = diag_k_idx + 1; l_idx < row_k_epos; l_idx++)
            {
                int l = bf_pair_j[l_idx];
                int idx_kl = k * nbf + l;
                double D_kl = D_mat[idx_kl] * 2.0;
                size_t offset = (size_t) l_idx * (size_t) df_nbf;
                double *df_tensor_row = df_tensor + offset;
                
                #pragma simd
                for (size_t p = 0; p < df_nbf; p++)
                    temp_J_thread[p] += D_kl * df_tensor_row[p];
            }
        }
        
        #pragma omp barrier
        reduce_temp_J(TinySCF->temp_J, temp_J_thread, TinySCF->df_nbf_16, tid, TinySCF->nthreads);
        
        #pragma omp master
        t1 = get_wtime_sec();

        // Build J matrix
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < nbf; i++)
        {
            int diag_i_idx = bf_pair_diag[i];
            int row_i_epos = bf_mask_displs[i + 1];
            for (int j_idx = diag_i_idx; j_idx < row_i_epos; j_idx++)
            {
                int j = bf_pair_j[j_idx];
                
                size_t offset = (size_t) j_idx * (size_t) df_nbf;
                double *df_tensor_row = df_tensor + offset;
                
                double t = 0;
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

static void set_batch_dgemm_arrays_K(TinySCF_t TinySCF)
{
    int nbf    = TinySCF->nbasfuncs;
    int df_nbf = TinySCF->df_nbf;
    int n_occ  = TinySCF->n_occ;
    int mat_K_BS = TinySCF->mat_K_BS;
    int nblocks0 = nbf / mat_K_BS;
    int last_block_size = nbf % mat_K_BS;
    int *group_size = TinySCF->mat_K_group_size;
    group_size[0] = (nblocks0 * (nblocks0 + 1)) / 2;
    if (last_block_size > 0)
    {
        group_size[1] = nblocks0;
        group_size[2] = 1;
    } else {
        group_size[1] = 0;
        group_size[2] = 0;
    }
    
    int cnt0 = 0, cnt1 = group_size[0];
    int cnt2 = group_size[0] + group_size[1];
    for (int i = 0; i < nbf; i += mat_K_BS)
    {
        int i_len = mat_K_BS < (nbf - i) ? mat_K_BS : (nbf - i);
        for (int j = i; j < nbf; j += mat_K_BS)
        {
            int j_len = mat_K_BS < (nbf - j) ? mat_K_BS : (nbf - j);
            
            size_t offset_i0 = (size_t) i * (size_t) n_occ * (size_t) df_nbf;
            size_t offset_j0 = (size_t) j * (size_t) n_occ * (size_t) df_nbf;
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
            TinySCF->mat_K_k[gid]      = n_occ * df_nbf;
            TinySCF->mat_K_alpha[gid]  = 1.0;
            TinySCF->mat_K_beta[gid]   = 0.0;
            TinySCF->mat_K_a[cnt]      = temp_K_i;
            TinySCF->mat_K_b[cnt]      = temp_K_j;
            TinySCF->mat_K_c[cnt]      = K_ij;
            TinySCF->mat_K_lda[gid]    = n_occ * df_nbf;
            TinySCF->mat_K_ldb[gid]    = n_occ * df_nbf;
            TinySCF->mat_K_ldc[gid]    = nbf;
        }
    }
}

static void set_batch_dgemm_arrays_temp_K(TinySCF_t TinySCF)
{
    int nbf    = TinySCF->nbasfuncs;
    int df_nbf = TinySCF->df_nbf;
    int n_occ  = TinySCF->n_occ;
    int *bf_mask_displs = TinySCF->bf_mask_displs;
    double *Cocc_tmp  = TinySCF->pqA;
    double *df_tensor = TinySCF->df_tensor;
    double *temp_K    = TinySCF->temp_K;
    
    for (int i = 0; i < nbf; i++)
    {
        int row_spos = bf_mask_displs[i];
        int row_epos = bf_mask_displs[i + 1];
        int row_len  = row_epos - row_spos;
        
        size_t offset_a = (size_t) row_spos * (size_t) df_nbf;
        size_t offset_b = (size_t) row_spos * (size_t) df_nbf;
        size_t offset_c = (size_t) i * (size_t) n_occ * (size_t) df_nbf;
        double *A_ptr = Cocc_tmp  + offset_a;
        double *B_ptr = df_tensor + offset_b;
        double *C_ptr = temp_K    + offset_c;
        
        TinySCF->mat_K_transa[i] = CblasTrans;
        TinySCF->mat_K_transb[i] = CblasNoTrans;
        TinySCF->mat_K_m[i]      = n_occ;
        TinySCF->mat_K_n[i]      = df_nbf;
        TinySCF->mat_K_k[i]      = row_len;
        TinySCF->mat_K_alpha[i]  = 1.0;
        TinySCF->mat_K_beta[i]   = 0.0;
        TinySCF->mat_K_a[i]      = A_ptr;
        TinySCF->mat_K_b[i]      = B_ptr;
        TinySCF->mat_K_c[i]      = C_ptr;
        TinySCF->mat_K_lda[i]    = df_nbf;
        TinySCF->mat_K_ldb[i]    = df_nbf;
        TinySCF->mat_K_ldc[i]    = df_nbf;
        TinySCF->mat_K_group_size[i] = 1;
    }
}

// Generate the temporary tensor for K matrix and form K matrix using D matrix
// High flop-per-byte ratio: access: nbf * df_nbf * (nbf + n_occ) , compute: nbf^2 * df_nbf * n_occ
// Note: the K_mat is not completed, the symmetrizing is done later
static void build_K_mat_Cocc(TinySCF_t TinySCF, double *temp_K_t, double *K_mat_t)
{
    double *K_mat     = TinySCF->K_mat;
    double *df_tensor = TinySCF->df_tensor;
    double *temp_K    = TinySCF->temp_K;
    double *Cocc_tmp  = TinySCF->pqA;
    double *Cocc_mat  = TinySCF->Cocc_mat;
    int nbf    = TinySCF->nbasfuncs;
    int df_nbf = TinySCF->df_nbf;
    int n_occ  = TinySCF->n_occ;
    
    double t0, t1, t2;
    
    t0 = get_wtime_sec();
    
    // Construct temporary tensor for K matrix
    // Formula: temp_K(i, s, p) = dot(Cocc_mat(1:nbf, s), df_tensor(i, 1:nbf, p))
    set_batch_dgemm_arrays_temp_K(TinySCF);
    int ngroups_temp_K = nbf;
    int *bf_pair_j = TinySCF->bf_pair_j;
    int *bf_mask_displs = TinySCF->bf_mask_displs;
    int bf_pair_cnt = TinySCF->bf_mask_displs[nbf];
    #pragma omp parallel for schedule(dynamic)
    for (int irow = 0; irow < bf_pair_cnt; irow++)
    {
        int j = bf_pair_j[irow];
        memcpy(Cocc_tmp + irow * df_nbf, Cocc_mat + j * n_occ, DBL_SIZE * n_occ);
    }
    cblas_dgemm_batch(
        CblasRowMajor, TinySCF->mat_K_transa, TinySCF->mat_K_transb,
        TinySCF->mat_K_m, TinySCF->mat_K_n, TinySCF->mat_K_k,
        TinySCF->mat_K_alpha,  
        (const double **) TinySCF->mat_K_a, TinySCF->mat_K_lda,
        (const double **) TinySCF->mat_K_b, TinySCF->mat_K_ldb,
        TinySCF->mat_K_beta,
        TinySCF->mat_K_c, TinySCF->mat_K_ldc,
        ngroups_temp_K, TinySCF->mat_K_group_size
    );
    /*
    double *A_ptr  = TinySCF->Cocc_mat;
    double *temp_A = TinySCF->temp_K_A;
    int    *bf_pair_j      = TinySCF->bf_pair_j;
    int    *bf_mask_displs = TinySCF->bf_mask_displs;
    for (int i = 0; i < nbf; i++)
    {
        size_t offset_c = (size_t) i * (size_t) n_occ * (size_t) df_nbf;
        double *C_ptr = temp_K + offset_c;
        
        int j_idx_spos = bf_mask_displs[i];
        int j_idx_epos = bf_mask_displs[i + 1];
        for (int j_idx = j_idx_spos; j_idx < j_idx_epos; j_idx++)
        {
            int j = bf_pair_j[j_idx];
            int cnt = j_idx - j_idx_spos;
            memcpy(temp_A + cnt * n_occ,  A_ptr + j * n_occ,  DBL_SIZE * n_occ);
        }
        
        int ncols = j_idx_epos - j_idx_spos;
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            n_occ, df_nbf, ncols,
            1.0, temp_A, n_occ, df_tensor + j_idx_spos * df_nbf, df_nbf,
            0.0, C_ptr, df_nbf
        );
    }
    */
    
    t1 = get_wtime_sec();

    // Build K matrix
    // Formula: K(i, j) = sum_{s=1}^{n_occ} [ dot(temp_K(i, s, 1:df_nbf), temp_K(j, s, 1:df_nbf)) ]
    if (nbf <= 1024)
    {    
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            nbf, nbf, n_occ * df_nbf, 
            1.0, temp_K, n_occ * df_nbf, temp_K, n_occ * df_nbf,
            0.0, K_mat, nbf
        );
    } else {
        int ngroups = 3;
        set_batch_dgemm_arrays_K(TinySCF);
        if (TinySCF->mat_K_group_size[1] == 0) ngroups = 1;
        cblas_dgemm_batch(
            CblasRowMajor, TinySCF->mat_K_transa, TinySCF->mat_K_transb,
            TinySCF->mat_K_m, TinySCF->mat_K_n, TinySCF->mat_K_k,
            TinySCF->mat_K_alpha,  
            (const double **) TinySCF->mat_K_a, TinySCF->mat_K_lda,
            (const double **) TinySCF->mat_K_b, TinySCF->mat_K_ldb,
            TinySCF->mat_K_beta,
            TinySCF->mat_K_c, TinySCF->mat_K_ldc,
            ngroups, TinySCF->mat_K_group_size
        );
    }
    
    t2 = get_wtime_sec();
    
    *temp_K_t = t1 - t0;
    *K_mat_t  = t2 - t1;
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
    build_K_mat_Cocc(TinySCF, &temp_K_t, &K_mat_t);
    
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
