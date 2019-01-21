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
    //double *df_tensor = TinySCF->df_tensor;
    double *temp_J    = TinySCF->temp_J;
    int nbf           = TinySCF->nbasfuncs;
    int df_nbf        = TinySCF->df_nbf;
    int nthreads      = TinySCF->nthreads;
    int *bf_pair_mask = TinySCF->bf_pair_mask;
    double *df_tensor0 = TinySCF->df_tensor0;
    
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
            //size_t offset = (size_t) (k * nbf + k) * (size_t) df_nbf;
            //double *df_tensor_row = df_tensor + offset;
            size_t offset = (size_t) diag_k_idx * (size_t) df_nbf;
            double *df_tensor_row = df_tensor0 + offset;
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
                //size_t offset = (size_t) (l * nbf + k) * (size_t) df_nbf;
                //double *df_tensor_row = df_tensor + offset;
                size_t offset = (size_t) l_idx * (size_t) df_nbf;
                double *df_tensor_row = df_tensor0 + offset;
                
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
                //size_t offset = (size_t) (i * nbf + j) * (size_t) df_nbf;
                //double *df_tensor_row = df_tensor + offset;
                
                size_t offset = (size_t) j_idx * (size_t) df_nbf;
                double *df_tensor_row = df_tensor0 + offset;
                
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


static void set_batch_dgemm_arrays_D(TinySCF_t TinySCF)
{
    int nbf    = TinySCF->nbasfuncs;
    int df_nbf = TinySCF->df_nbf;
    int n_occ  = TinySCF->n_occ;
    
    for (int i = 0; i < nbf; i++)
    {
        TinySCF->temp_K_a[i] = TinySCF->D_mat;
        TinySCF->temp_K_b[i] = TinySCF->df_tensor + i * df_nbf;
        TinySCF->temp_K_c[i] = TinySCF->temp_K    + i * df_nbf;
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
    
    double *df_tensor0 = TinySCF->df_tensor0;
    
    // TODO: Optimize this transpose
    double *DT = (double*) malloc(DBL_SIZE * nbf * nbf);
    #pragma omp parallel for
    for (int i = 0; i < nbf; i++)
    {
        double *DT_row = DT + i * nbf;
        double *D_mat_col = D_mat + i;
        #pragma omp simd
        for (int j = 0; j < nbf; j++)
            DT_row[j] = D_mat_col[j * nbf];
    }
    
    // Construct temporary tensor for K matrix
    // Formula: temp_K(k, j, p) = dot(D_mat(k, 1:nbf), df_tensor(1:nbf, j, p))
    int *bf_pair_mask = TinySCF->bf_pair_mask;
    double *temp_K_A = TinySCF->temp_K_A;
    double *temp_K_B = TinySCF->temp_K_B;
    for (int j = 0; j < nbf; j++)
    {
        size_t offset_c = (size_t) j * (size_t) df_nbf;
        size_t ldC = (size_t) nbf * (size_t) df_nbf;
        double *C_ptr = temp_K + offset_c;
        //double *B_ptr = df_tensor + offset_c;
        
        int cnt = 0;
        for (int l = 0; l < nbf; l++)
        {
            int bf_pair_idx = bf_pair_mask[l * nbf + j];
            if (bf_pair_idx >= 0)
            {
                memcpy(temp_K_A + cnt * nbf, DT + l * nbf, DBL_SIZE * nbf);
                //memcpy(temp_K_B + cnt * df_nbf, B_ptr + l * ldC, DBL_SIZE * df_nbf);
                memcpy(temp_K_B + cnt * df_nbf, df_tensor0 + bf_pair_idx * df_nbf, DBL_SIZE * df_nbf);
                cnt++;
            }
        }
        
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            nbf, df_nbf, cnt,
            1.0, temp_K_A, nbf, temp_K_B, df_nbf, 
            0.0, C_ptr, ldC
        );
    }
    
    free(DT);
    
    t1 = get_wtime_sec();
    
    // Build K matrix
    // Formula: K(i, j) = sum_{k=1}^{nbf} [ dot(df_tensor(i, k, 1:df_nbf), temp_K(k, j, 1:df_nbf)) ]
    memset(K_mat, 0, DBL_SIZE * nbf * nbf);
    int ngroups = 3;
    if (TinySCF->mat_K_group_size[1] == 0) ngroups = 1;
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
            ngroups, TinySCF->mat_K_group_size
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

static void set_batch_dgemm_arrays_Cocc(TinySCF_t TinySCF)
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

// Generate the temporary tensor for K matrix and form K matrix using D matrix
// High flop-per-byte ratio: access: nbf * df_nbf * (nbf + n_occ) , compute: nbf^2 * df_nbf * n_occ
// Note: the K_mat is not completed, the symmetrizing is done later
static void build_K_mat_Cocc(TinySCF_t TinySCF, double *temp_K_t, double *K_mat_t)
{
    double *K_mat = TinySCF->K_mat;
    int nbf       = TinySCF->nbasfuncs;
    int df_nbf    = TinySCF->df_nbf;
    int n_occ     = TinySCF->n_occ;
    
    double t0, t1, t2;
    
    t0 = get_wtime_sec();
    
    double *df_tensor0 = TinySCF->df_tensor0;
    
    // Construct temporary tensor for K matrix
    // Formula: temp_K(s, i, p) = dot(Cocc_mat(1:nbf, s), df_tensor(i, 1:nbf, p))
    double *A_ptr  = TinySCF->Cocc_mat;
    double *temp_A = TinySCF->temp_K_A;
    double *temp_B = TinySCF->temp_K_B;
    int    *bf_pair_mask   = TinySCF->bf_pair_mask;
    int    *bf_pair_j      = TinySCF->bf_pair_j;
    int    *bf_mask_displs = TinySCF->bf_mask_displs;
    const int temp_K_m   = n_occ;
    const int temp_K_n   = df_nbf;
    const int temp_K_k   = nbf;
    const int temp_K_lda = n_occ;
    const int temp_K_ldb = df_nbf;
    const int temp_K_ldc = df_nbf * nbf;
    for (int i = 0; i < nbf; i++)
    {
        size_t offset_b = (size_t) i * (size_t) nbf * (size_t) df_nbf;
        size_t offset_c = (size_t) i * (size_t) df_nbf;
        //double *B_ptr = TinySCF->df_tensor + offset_b;
        double *C_ptr = TinySCF->temp_K    + offset_c;
        
        int cnt = 0;
        int nbf_i = i * nbf;
        
        int j_idx_spos = bf_mask_displs[i];
        int j_idx_epos = bf_mask_displs[i + 1];
        for (int j_idx = j_idx_spos; j_idx < j_idx_epos; j_idx++)
        {
            int j = bf_pair_j[j_idx];
            memcpy(temp_A + cnt * n_occ,  A_ptr + j * n_occ,  DBL_SIZE * n_occ);
            //memcpy(temp_B + cnt * df_nbf, B_ptr + j * df_nbf, DBL_SIZE * df_nbf);
            cnt++;
        }
        
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            n_occ, df_nbf, cnt,
            //1.0, temp_A, n_occ, temp_B, df_nbf,
            1.0, temp_A, n_occ, df_tensor0 + j_idx_spos * df_nbf, df_nbf,
            0.0, C_ptr, temp_K_ldc
        );
    }
    
    t1 = get_wtime_sec();

    // Build K matrix
    // Formula: K(i, j) = sum_{s=1}^{n_occ} [ dot(temp_K(s, i, 1:df_nbf), temp_K(s, j, 1:df_nbf)) ]
    memset(K_mat, 0, DBL_SIZE * nbf * nbf);
    int ngroups = 3;
    if (TinySCF->mat_K_group_size[1] == 0) ngroups = 1;
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
            ngroups, TinySCF->mat_K_group_size
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
        set_batch_dgemm_arrays_D(TinySCF);
        build_K_mat_D(TinySCF, &temp_K_t, &K_mat_t);
        set_batch_dgemm_arrays_Cocc(TinySCF);
    } else {
        build_K_mat_Cocc(TinySCF, &temp_K_t, &K_mat_t);
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
