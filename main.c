#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#include "CMS.h"
#include "TinySCF.h"
#include "TinySCF_init_free.h"
#include "build_DF_tensor.h"

static void print_usage(char *exe_name)
{
	printf("Usage: %s <basis> <denfit_basis> <xyz> <niter>\n", exe_name);
}

int main(int argc, char **argv)
{
	if (argc < 5)
	{
		print_usage(argv[0]);
		return 255;
	}
	
	TinySCF_t TinySCF;
	TinySCF = (TinySCF_t) malloc(sizeof(struct TinySCF_struct));
	assert(TinySCF != NULL);
	
	init_TinySCF(TinySCF, argv[1], argv[2], argv[3], atoi(argv[4]));

	TinySCF_init_batch_dgemm_arrays(TinySCF);
	
	TinySCF_compute_Hcore_Ovlp_mat(TinySCF);
	
	TinySCF_compute_sq_Schwarz_scrvals(TinySCF);
	
	TinySCF_get_initial_guess(TinySCF);
	
	TinySCF_build_DF_tensor(TinySCF);

	TinySCF_do_SCF(TinySCF);

	TinySCF_free_batch_dgemm_arrays(TinySCF);
	
	free_TinySCF(TinySCF);
	
	return 0;
}
