#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#include "CMS.h"
#include "TinySCF.h"

static void print_usage(char *exe_name)
{
	printf("Usage: %s <basis> <xyz> <niter>\n", exe_name);
}

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		print_usage(argv[0]);
		return 255;
	}
	
	TinySCF_t TinySCF;
	TinySCF = (TinySCF_t) malloc(sizeof(struct TinySCF_struct));
	assert(TinySCF != NULL);
	
	init_TinySCF(TinySCF, argv[1], argv[2], atoi(argv[3]));
	
	TinySCF_compute_Hcore_Ovlp_mat(TinySCF);
	
	TinySCF_compute_sq_Schwarz_scrvals(TinySCF);
	
	TinySCF_get_initial_guess(TinySCF);
	
	TinySCF_do_SCF(TinySCF);
	
	free_TinySCF(TinySCF);
	
	return 0;
}
