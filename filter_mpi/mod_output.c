#include "mod_output.h"

/****************************************************************************/

void mod_output(
    double *psitot,
    xyz_st *R,
    double *eig_vals,
    double *sigma_E,
    grid_st *grid,
    index_st *ist,
    par_st *par,
    flag_st *flag,
    parallel_st *parallel)
{

  /************************************************************/
  /*******************  DECLARE VARIABLES   *******************/
  /************************************************************/

  FILE *ppsi;

  long jgrid;
  long jgrid_real;
  long jgrid_imag;
  long jmn;

  const int fgas = flag->getAllStates;
  long mn = ist->mn_states_tot;
  const long stlen = ist->complex_idx * ist->nspinngrid;
  const double secut = par->sigma_E_cut;
  long eig = 0;
  int *conv = NULL;

  // Output file names carry the spin tag (par->file_tag) so each spin channel of a
  // spin-polarized run writes its own eval/psi/output files. For a normal run
  // file_tag is "", giving the historical names eval.dat / psi.dat / output.dat.
  char eval_name[32];
  char psi_name[32];
  char output_name[32];
  snprintf(eval_name, sizeof(eval_name), "eval%s.dat", par->file_tag);
  snprintf(psi_name, sizeof(psi_name), "psi%s.dat", par->file_tag);
  snprintf(output_name, sizeof(output_name), "output%s.dat", par->file_tag);

  /************************************************************/
  /*******************   WRITE ALL STATES   *******************/
  /************************************************************/
  /*****  write the eigenvals/states to eval.dat/psi.dat  *****/
  /************************************************************/

  if (fgas == 1)
  {
    printf(
        "\ngetAllStates flag on\nWriting all eigenstates to disk\n");

    write_eval_dat(eig_vals, sigma_E, mn, eval_name);

    write_psi_dat(psitot, mn * stlen, psi_name);
  }

  /************************************************************/
  /******************    WRITE CONV STATES   ******************/
  /************************************************************/

  // Write output for only converged eigenstates
  else
  {
    printf(
        "\ngetAllStates flag off\nWriting only converged eigenstates to disk\n");
    // Select the converged states
    ALLOCATE(&conv, mn, "conv in mod_output");

    mn = select_conv_states(sigma_E, conv, secut, mn);

    // Reorder eig_vals and write eval.dat
    for (jmn = 0; jmn < mn; jmn++)
    {

      // Get idx of converged state
      eig = conv[jmn];

      // Make the jmn-th element of eig_vals equal to this value
      eig_vals[jmn] = eig_vals[eig];
      sigma_E[jmn] = sigma_E[eig];
    }

    //
    write_eval_dat(eig_vals, sigma_E, mn, eval_name);
    //

    // Write the converged psi.dat
    ppsi = fopen(psi_name, "w");

    if (ppsi == NULL)
    {
      fprintf(stderr, "ERROR opening psi.dat\n");
      exit(EXIT_FAILURE);
    }

    for (jmn = 0; jmn < mn; jmn++)
    {
      eig = conv[jmn];

      // // Move file ptr to write eigenstates contiguously in psi.dat
      // fseek(ppsi, jmn * stlen * sizeof(double), SEEK_SET);

      // Write the eig'th eigenstate to psi.dat psitot
      fwrite(&psitot[eig * stlen], sizeof(double), stlen, ppsi);

      // Modify psitot to only contain converged eigenstates
      if (1 == flag->isComplex)
      {
        for (jgrid = 0; jgrid < ist->nspinngrid; jgrid++)
        {
          jgrid_real = ist->complex_idx * jgrid;
          jgrid_imag = ist->complex_idx * jgrid + 1;

          psitot[jmn * stlen + jgrid_real] = psitot[eig * stlen + jgrid_real];
          psitot[jmn * stlen + jgrid_imag] = psitot[eig * stlen + jgrid_imag];
        }
      }
      else
      {
        for (jgrid = 0; jgrid < ist->nspinngrid; jgrid++)
        {
          psitot[jmn * stlen + jgrid] = psitot[eig * stlen + jgrid];
        }
      }
    }
    fclose(ppsi);

    printf("No. of eigenstates with sigE < %.2lg: new_mstot = %ld\n", secut, mn);
  }

  /************************************************************/
  /*******************    RESTART FILTER    *******************/
  /************************************************************/

  // If no eigenstates were obtained, then the job has failed
  // alternate options should be pursued
  if (mn == 0)
  {
    printf("The filter diagonalization yielded %ld eigenstates\n", mn);
    if ((flag->retryFilter == 1) && (flag->alreadyTried == 0))
    {
      ist->n_filter_cycles += 16;
      ist->ncheby += 1024;
      printf("New ns = %ld, new ncheby = %ld\n", ist->n_filter_cycles, ist->ncheby);
      flag->alreadyTried = 1; // prevent filter from repeating indefinitely
    }
    else
    {
      fprintf(stderr, "Filter yielded %ld eigenstates (T_T)\n", mn);
      fprintf(stderr, "Job failed :( exiting reluctantly\n");
      exit(EXIT_FAILURE);
    }
  }

  /************************************************************/
  /*******************   SAVE OUTPUT.DAT    *******************/
  /************************************************************/

  // Filter diagonalization procedure has concluded
  // Print output.dat for BSE unless explicitly NOT requested
  if (0 != flag->saveOutput)
  {
    save_output(
        output_name, psitot, eig_vals, sigma_E, R, grid, ist, par, flag, parallel);
  }

  return;
}

/****************************************************************************/

int select_conv_states(
    double *sigma_E,
    int *arr,
    double secut,
    int n_elems)
{

  int i;
  int cntr = 0;

  for (i = 0; i < n_elems; i++)
  {
    // If the sigma_E val > secut, skip
    if (sigma_E[i] > secut)
    {
      continue;
    }
    // If not, add index to arr and increment cntr
    else
    {
      arr[cntr] = i;
      cntr++;
    }
  }

  return cntr;
}

/****************************************************************************/
