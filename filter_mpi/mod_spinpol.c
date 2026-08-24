#include "mod_spinpol.h"

/*****************************************************************************/

void run_spinpol(
    zomplex*      psi,
    zomplex*      phi,
    double*       pot_local,
    pot_st*       pot,
    xyz_st*       R,
    atom_info*    atom,
    grid_st*      grid,
    zomplex*      LS,
    nlc_st*       nlc,
    long*         nl,
    zomplex*      an,
    double*       zn,
    double*       ene_targets,
    double*       ksqr,
    lattice_st*   lattice,
    vector*       G_vecs,
    vector*       k_vecs,
    double*       eig_vals,
    double*       sigma_E,
    index_st*     ist,
    par_st*       par,
    flag_st*      flag,
    parallel_st*  parallel)
{
  /*******************************************************************
   * Run the full non-periodic filter-diagonalization pipeline once   *
   * for each collinear spin channel (up, then down). The two channels *
   * share everything except the local potential and the output file   *
   * names, so we loop, rebuild the local potential for the channel,    *
   * and reuse the existing modules exactly as the non-periodic path    *
   * in main.c does (no restart / no periodic branches here -- those     *
   * are rejected up front in read_input for spinPolarized runs).        *
   ********************************************************************/

  const int mpir = parallel->mpi_rank;

  // Labels for the two collinear spin channels. spin index 0 -> up, 1 -> down.
  const char *spin_tag[2] = {"_up", "_dn"};
  const char *spin_label[2] = {"UP", "DOWN"};

  for (int spin = 0; spin < 2; spin++)
  {
    /************************************************************/
    /**************   SELECT THIS SPIN CHANNEL   ****************/
    /************************************************************/

    par->spin_channel = spin;
    // file_tag steers read_pot (pot<X>_up/_dn.par) and mod_output (eval_up/_dn.dat).
    strncpy(par->file_tag, spin_tag[spin], sizeof(par->file_tag) - 1);
    par->file_tag[sizeof(par->file_tag) - 1] = '\0';

    if (mpir == 0)
    {
      write_separation(stdout, "T");
      printf("\nSPIN-POLARIZED CHANNEL: SPIN %s | %s\n", spin_label[spin], get_time());
      write_separation(stdout, "B");
      fflush(stdout);
    }

    /************************************************************/
    /*************   BUILD THIS CHANNEL'S LOCAL POT   ***********/
    /************************************************************/
    /*** Non-local / spin-orbit projectors are spin-independent and ***/
    /*** were already built once by mod_pseudopot; only the local    ***/
    /*** potential differs between spin channels.                    ***/

    build_local_pot_channel(pot_local, pot, R, atom, grid, ist, par, flag, parallel);

    /************************************************************/
    /*******************  ALLOCATE FILTER MEM   *****************/
    /************************************************************/
    /*** psi_rank holds this rank's filtered states. It is allocated ***/
    /*** and freed inside the loop so each channel starts clean.     ***/

    double *psi_rank = NULL;
    ALLOCATE(&psi_rank, ist->psi_rank_size, "psi_rank in run_spinpol");

    double *psitot = NULL; // gather_mpi_filt allocates this on rank 0 only

    /************************************************************/
    /*******************      RUN FILTER      *******************/
    /************************************************************/

    mod_filter(
        psi_rank, psi, phi, pot_local, grid, LS, nlc, nl, an, zn,
        ene_targets, ksqr, lattice, G_vecs, k_vecs, ist, par, flag, parallel);

    /************************************************************/
    /*******************      GATHER STATES     *****************/
    /************************************************************/

    gather_mpi_filt(psi_rank, &psitot, ist, par, flag, parallel);
    free(psi_rank);
    psi_rank = NULL;

    /************************************************************/
    /*******************   ORTHOGONALIZE   *********************/
    /************************************************************/
    /*** A spin-polarized run is always a fresh filter (restart is    ***/
    /*** rejected in read_input), so orthogonalization uses the serial ***/
    /*** mod_ortho on rank 0 -- exactly like main.c's fresh-run path.   ***/
    /*** The distributed variant mod_portho is a restart-from-ortho     ***/
    /*** path that re-reads filtered states from disk, so it is not     ***/
    /*** applicable here. The heavy, MPI-distributed work is the filter ***/
    /*** above; the diagonalization below can still distribute across   ***/
    /*** ranks via MPIDiag.                                             ***/

    omp_set_num_threads(parallel->nthreads);

    if (mpir == 0)
    {
      mod_ortho(
          psitot, pot_local, grid, nlc, nl, an, zn, ene_targets,
          ksqr, ist, par, flag, parallel);

      psitot = realloc(psitot, ist->mn_states_tot * ist->nspinngrid * ist->complex_idx * sizeof(psitot[0]));
    }

    /************************************************************/
    /*******************    DIAGONALIZE    *********************/
    /************************************************************/
    /*** Called on all ranks (collective when MPIDiag is on); the    ***/
    /*** serial path is internally guarded to rank 0.                ***/

    mod_diag(
        psitot, pot_local, eig_vals, sigma_E, grid, G_vecs, k_vecs, LS, nlc, nl,
        an, zn, ene_targets, ksqr, ist, par, flag, parallel);

    /************************************************************/
    /****************   SIGMA / OUTPUT (rank 0)   ***************/
    /************************************************************/

    if (mpir == 0)
    {
      printf("Entering mod_sigma\n");
      fflush(0);

      mod_sigma(
          psitot, pot_local, eig_vals, sigma_E, grid, LS, nlc, nl,
          ksqr, ist, par, flag, parallel);

      write_separation(stdout, "T");
      printf("\nPOST-PROCESSING SPIN %s | %s\n", spin_label[spin], get_time());
      write_separation(stdout, "B");
      fflush(stdout);

      mod_output(psitot, R, eig_vals, sigma_E, grid, ist, par, flag, parallel);

      mod_optional_output(psitot, grid, ist, par, flag, parallel);

      free(psitot);
      psitot = NULL;
    }

    // Synchronize before rebuilding the next spin channel's potential so no rank
    // starts channel N+1 while rank 0 is still writing channel N's output.
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Restore the default (untagged) file naming for anything downstream.
  par->file_tag[0] = '\0';
  par->spin_channel = 0;

  return;
}

/*****************************************************************************/
