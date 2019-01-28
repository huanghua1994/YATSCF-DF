#ifndef _YATSCF_TFOCK_H_
#define _YATSCF_TFOCK_H_

#include <mkl.h>
#include <omp.h>
#include "CMS.h"

#define MAX_DIIS 10
#define MIN_DIIS 2

// Tiny SCF engine
struct TinySCF_struct 
{
    // OpenMP parallel setting and buffer
    int    nthreads;        // Number of threads
    int    max_buf_size;    // Maximum buffer size for each thread's accumulating Fock matrix
    double *Accum_Fock_buf; // Pointer to all thread's buffer for accumulating Fock matrix
    
    // Chemical system info
    BasisSet_t basis;       // Basis set object for storing chemical system info, handled by libCMS
    BasisSet_t df_basis;    // Basis set object for storing density fitting info, handled by libCMS
    int natoms, nshells;    // Number of atoms and shells
    int nbasfuncs, n_occ;   // Number of basis functions and occupied orbits
    int charge, electron;   // Charge and number of electrons 
    int df_nbf, df_nshells; // Number of basis functions and shells for density fitting
    
    // Auxiliary variables 
    int nshellpairs;        // Number of shell pairs        (== nshells * nshells)
    int num_uniq_sp;        // Number of unique shell pairs (== nshells * (nshells+1) / 2)
    int mat_size;           // Size of matrices             (== nbasfuncs * nbasfuncs)
    int max_dim;            // Maximum value of dim{M, N, P, Q}
    
    // SCF iteration info
    int    niters, iter;    // Maximum and current SCF iteration
    double nuc_energy;      // Nuclear energy
    double HF_energy;       // Hartree-Fock energy
    double ene_tol;         // SCF termination criteria for energy change
    
    // Shell quartet screening 
    double shell_scrtol2;   // Square of the shell screening tolerance
    double max_scrval;      // == max(fabs(sp_scrval(:)))
    double max_df_scrval;   // Maximum screening value of density fitting basis sets
    double *sp_scrval;      // Screening values (upper bound) of each shell pair
    double *bf_pair_scrval; // Screening values (ERI values) of each basis function pair
    int    *uniq_sp_lid;    // Left shell id of all unique shell pairs
    int    *uniq_sp_rid;    // Right shell id of all unique shell pairs
    double *df_sp_scrval;   // Square of screening values (upper bound) of each shell pair in density fitting
    int    *bf_pair_mask;   // If a basis function pair survives the Schwarz screening
    int    *bf_pair_j;      // j of basis function pair (i, j) that survives screening
    int    *bf_pair_diag;   // Index of basis function pair (i, i) in all basis function pairs 
    int    *bf_mask_displs; // How many basis function pairs in (i, :) survive screening and their storing order
    int    num_bf_pair_scr; // Total number of basis function pairs that survive screening
    
    // ERIs
    Simint_t simint;        // Simint object for ERI, handled by libCMS
    int *shell_bf_sind;     // Index of the first basis function of each shell
    int *shell_bf_num;      // Number of basis function in each shell
    int *df_shell_bf_sind;  // Index of the first basis function of each shell in density fitting
    int *df_shell_bf_num;   // Number of basis function in each shell in density fitting
    
    // Matrices and temporary arrays in SCF
    double *Hcore_mat;      // Core Hamiltonian matrix
    double *S_mat;          // Overlap matrix
    double *F_mat;          // Fock matrix
    double *D_mat;          // Density matrix
    double *J_mat;          // Coulomb matrix 
    double *K_mat;          // Exchange matrix
    double *X_mat;          // Basis transformation matrix
    double *Cocc_mat;       // Temporary matrix for building density matrix
    double *eigval;         // Eigenvalues for building density matrix
    int    *ev_idx;         // Index of eigenvalues, for sorting
    double *tmp_mat;        // Temporary matrix
    
    // Density fitting tensors and buffers
    double *pqA, *Jpq, *df_tensor;
    double *temp_J, *temp_K;
    int    df_nbf_16;
    
    // Matrices and arrays for DIIS
    double *F0_mat;       // Previous X^T * F * X matrices
    double *R_mat;        // "Residual" matrix
    double *B_mat;        // Linear system coefficient matrix in DIIS
    double *FDS_mat;      // F * D * S matrix in Commutator DIIS
    double *DIIS_rhs;     // Linear system right-hand-side vector in DIIS
    int    *DIIS_ipiv;    // Permutation info for DGESV in DIIS
    int    DIIS_len;      // Number of previous F matrices
    int    DIIS_bmax_id;  // The ID of a previous F matrix whose residual has the largest 2-norm
    double DIIS_bmax;     // The largest 2-norm of the stored F matrices' residuals
    
    // MKL batch dgemm arrays
    int mat_K_BS, *mat_K_group_size, mat_K_ntiles;
    CBLAS_TRANSPOSE *mat_K_transa, *mat_K_transb;
    int *mat_K_m,   *mat_K_n,   *mat_K_k;
    int *mat_K_lda, *mat_K_ldb, *mat_K_ldc;
    double *mat_K_alpha, *mat_K_beta;
    double **mat_K_a, **mat_K_b, **mat_K_c;
    
    // Statistic 
    double mem_size, init_time, S_Hcore_time, shell_scr_time;
};

typedef struct TinySCF_struct* TinySCF_t;

// Initialize TinySCF with two Cartesian basis set file (.gbs format), a molecule 
// coordinate file and the number of SCF iterations (handled by libcint), and
// allocate all memory for other calculation
void init_TinySCF(
    TinySCF_t TinySCF, char *bas_fname, char *df_bas_fname, 
    char *xyz_fname, const int niters
);

// Destroy TinySCF, free all allocated memory
void free_TinySCF(TinySCF_t TinySCF);

// Allocate arrays for MKL batch gemm and precompute tile info
// for building the K matrix
void TinySCF_init_batch_dgemm_arrays(TinySCF_t TinySCF);

// Free MKL batch dgemm arrays
void TinySCF_free_batch_dgemm_arrays(TinySCF_t TinySCF);

// Compute core Hamiltonian and overlap matrix, and generate basis transform matrix
// The overlap matrix is not needed after generating basis transform matrix, and its
// memory space will be used as temporary space in other procedure
void TinySCF_compute_Hcore_Ovlp_mat(TinySCF_t TinySCF);

// Compute the screening values of each shell quartet and the unique shell pairs
// that survive screening using Schwarz inequality
void TinySCF_compute_sq_Schwarz_scrvals(TinySCF_t TinySCF);

// Prepare to use the sparsity of regular basis set shell pairs
void TinySCF_prepare_sparsity(TinySCF_t TinySCF);

// Generate initial guess for density matrix using SAD data (handled by libcint)
void TinySCF_get_initial_guess(TinySCF_t TinySCF);

// Perform SCF iterations
void TinySCF_do_SCF(TinySCF_t TinySCF);

#endif
