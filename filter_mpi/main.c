#include "main.h"

/*****************************************************************************/

int main(int argc, char *argv[])
{
  /*****************************************************************
   * This is the main function for Filter.x. It is the driver that  *
   * controls memory allocation, structs, and the filter algorithm. *
   * The algorithm computes quasiparticle excited states using      *
   * sparse-matrix techniques. It is applied to nanocrystal         *
   * systems through the use of semiempirical pseudopotentials.     *
   ******************************************************************/

  /*****************************************************************/
  /*************   DECLARE VARIABLES AND STRUCTS   *****************/
  /*****************************************************************/

  // zomplex types
  zomplex *psi;       // Wavefunction on the grid
  zomplex *phi;       // Wavefunction on the grid (helper)
  zomplex *an;        // Newton interpolation coefficients
  zomplex *LS = NULL; // Spin-orbit matrix

  // custom structs
  index_st ist;         // Indexes for the program
  par_st par;           // Parameters for the program
  flag_st flag;         // Flags for options in the program
  parallel_st parallel; // MPI parallelization information
  xyz_st *R;            // Atomic positions in the x, y, and z
  atom_info *atom;      // Atom specific information
  pot_st pot;           // Atomic pseudopotential information
  grid_st grid;         // Grid params and the real space grid
  grid.x = NULL;
  grid.y = NULL;
  grid.z = NULL;
  nlc_st *nlc = NULL; // Non-local pseudopotential info

  lattice_st lattice;
  vector *G_vecs = NULL;
  vector *k_vecs = NULL;
  // gauss_st*       gauss;          // Gaussian basis coeffs/exps

  // double arrays
  double *psitot = NULL; // All filtered states
  double *psi_rank;      // Filter states for each rank
  double *pot_local;     // Local pseudopotential on the grid
  double *SO_projectors; // Spin-orbit projectors
  double *ksqr;          // Kinetic energy sqr on the grid
  double *zn;            // Chebyshev support points
  double *eig_vals;      // Quasiparticle energies
  double *ene_targets;   // Energy targets for filtering
  double *sigma_E;       // Standard dev. of the energies

  // long int arrays and counters
  long *nl = NULL;

  // MPI Initialization

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &parallel.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &parallel.mpi_size);

  const int mpir = parallel.mpi_rank;
  parallel.mpi_root = 0;

  // k-point grouping (periodic path) is configured later in setup_k_communicators.
  // Default to a null communicator / single trivial group so the non-periodic path
  // and end-of-run cleanup are always well-defined.
  parallel.k_comm = MPI_COMM_NULL;
  parallel.k_color = 0;
  parallel.k_rank = parallel.mpi_rank;
  parallel.k_size = parallel.mpi_size;
  parallel.n_k_groups = 1;
  parallel.n_k_local = 0;
  parallel.k_global_start = 0;
  parallel.k_local_to_global = NULL;

  /************************************************************/
  /*******************   JOB PRE-PRINTING   *******************/
  /************************************************************/

  // Clock/Wall time output and stdout formatting
  time_t start_time = time(NULL); // Get the actual time for total wall runtime
  time_t start_clock = clock();   // Get the starting CPU clock time for total CPU runtime

  char *top;
  char *bottom;
  top = malloc(2 * sizeof(top[0]));
  bottom = malloc(2 * sizeof(bottom[0]));
  strcpy(top, "T\0");
  strcpy(bottom, "B\0");

  if (mpir == 0)
  {
    printf("******************************************************************************\n");
    printf("\nRUNNING PROGRAM: FILTER DIAGONALIZATION\n");
    printf("This calculation began at: %s\n", get_time());
    printf("Printing from root node %d", mpir);
    write_separation(stdout, bottom);
    fflush(stdout);
  }

  /************************************************************/
  /*******************   RUN INIT MODULE    *******************/
  /************************************************************/

  // allocate mem for all the atom types as a pre-step
  ALLOCATE(&(ist.atom_types), N_MAX_ATOM_TYPES, "ist->atom_types");

  mod_init(
      &grid, &(grid.x), &(grid.y), &(grid.z), &R, &atom, &ene_targets,
      &ksqr, &lattice, &G_vecs, &k_vecs, &ist, &par, &flag, &parallel);

  /************************************************************/
  /*******************    RUN MEM MODULE    *******************/
  /************************************************************/

  mod_mem_alloc(
      &psi_rank, &psi, &phi, &pot_local, &LS, &nlc, &nl, &SO_projectors,
      &an, &zn, &eig_vals, &sigma_E, &ist, &par, &flag, &parallel);

  /************************************************************/
  /******************   RUN PSEUDOPOT MODULE  *****************/
  /************************************************************/

  mod_pseudopot(
      pot_local, &pot, R, atom, &grid, LS, nlc, nl, SO_projectors,
      &ist, &par, &flag, &parallel);

  // Collinear spin-polarized runs build two independent scalar Hamiltonians
  // (spin up / spin down) that differ only in their local potential, and run the
  // full non-periodic filter->ortho->diag->sigma->output pipeline once per spin
  // channel, writing eval_up.dat / eval_dn.dat. This path is mutually exclusive
  // with the checkpoint/restart switch below (rejected in read_input), so it is
  // handled here and the restart switch is skipped.
  if (1 == flag.spinPolarized)
  {
    // The psi_rank buffer allocated by mod_mem_alloc is unused here; run_spinpol
    // allocates and frees its own per-channel buffer.
    free(psi_rank);
    psi_rank = NULL;

    run_spinpol(
        psi, phi, pot_local, &pot, R, atom, &grid, LS, nlc, nl, an, zn,
        ene_targets, ksqr, &lattice, G_vecs, k_vecs, eig_vals, sigma_E,
        &ist, &par, &flag, &parallel);
  }
  else
  {
  // This code supports restarting the job from a saved state.
  // See save.c for formatting of checkpoint files.
  // See read_input in read.c for specifying the checkpoint restart
  switch (flag.restartFromCheckpoint)
  {

  // No restart requested
  case 0:

    // // If the job uses a Gaussian basis, run the job in the Gaussian basis
    // if (1 == flag.useGaussianBasis){
    //   //
    //   gauss_driver(
    //     gauss, pot_local, eig_vals, R, atom,
    //     &grid, &ist, &par, &flag, &parallel);
    //   //
    //   exit(0);
    // }

    /************************************************************/
    /*******************  RUN FILTER MODULE   *******************/
    /************************************************************/

    mod_filter(
        psi_rank, psi, phi, pot_local, &grid, LS, nlc, nl, an, zn,
        ene_targets, ksqr, &lattice, G_vecs, k_vecs, &ist, &par, &flag, &parallel);

    if (1 == flag.calcFilterOnly)
    {
      exit(0);
    }

    // Periodic path: each k-group gathers, orthogonalizes, diagonalizes and writes
    // its k-points' eigenvalues independently. This replaces the gather/ortho/diag/
    // sigma/output modules below, so we exit the restart switch once it returns.
    if (1 == flag.periodic)
    {
      run_periodic_postfilter(
          psi_rank, pot_local, R, &grid, G_vecs, k_vecs, LS, nlc, nl,
          ksqr, eig_vals, sigma_E, &ist, &par, &flag, &parallel);
      free(psi_rank);
      break;
    }

    // Gather all psi_rank from MPI ranks into psitot
    gather_mpi_filt(psi_rank, &psitot, &ist, &par, &flag, &parallel);
    free(psi_rank);

    // Save a checkpoint if requested
    if ((1 == flag.saveCheckpoints) && (0 == mpir))
    {
      save_job_state(
          "checkpoint_1.dat", par.checkpoint_id, psitot, pot_local, ksqr,
          an, zn, ene_targets, nl, nlc, &grid, &ist, &par, &flag, &parallel);
    }

  // Restart from orthogonalization module
  case 1:
    omp_set_num_threads(parallel.nthreads);

    /************************************************************/
    /*******************   RUN ORTHO MODULE   *******************/
    /************************************************************/
    // Periodic restartFromOrtho: skip the filter step and read each
    // k-point's filtered states back from psi-filt-[k].dat, then run the
    // per-k ortho/diag/sigma/output pipeline. Mirrors run_periodic_postfilter
    // (case 0); replaces the non-periodic ortho/diag below, so we break out
    // of the restart switch once it returns.
    if ((1 == flag.restartFromOrtho) && (1 == flag.periodic))
    {
      run_periodic_restart_from_ortho(
          psi_rank, pot_local, R, &grid, G_vecs, k_vecs, LS, nlc, nl,
          ksqr, eig_vals, sigma_E, &ist, &par, &flag, &parallel);
      free(psi_rank);
      break;
    }

    if ((1 == flag.restartFromOrtho) && (1 == flag.MPIOrtho))
    {
      mod_portho(
          &psitot, &psi_rank, pot_local, eig_vals, sigma_E, R, LS, nlc, nl,
          ksqr, &grid, &ist, &par, &flag, &parallel);
    }

    if (mpir == 0) // subsequent modules performed on a single rank
    {

      /************************************************************/
      /*******************  RESTART FROM ORTHO  *******************/
      /************************************************************/

      if ((1 == flag.restartFromOrtho) && (0 == flag.MPIOrtho))
      {
        restart_from_ortho(&psitot, &ist, &par, &flag, &parallel);
      }

      mod_ortho(
          psitot, pot_local, &grid, nlc, nl, an, zn, ene_targets,
          ksqr, &ist, &par, &flag, &parallel);

      psitot = realloc(psitot, ist.mn_states_tot * ist.nspinngrid * ist.complex_idx * sizeof(psitot[0]));

      // Save checkpoint if requested
      if (1 == flag.saveCheckpoints)
      {
        save_job_state(
            "checkpoint_2.dat", par.checkpoint_id, psitot, pot_local, ksqr,
            an, zn, ene_targets, nl, nlc, &grid, &ist, &par, &flag, &parallel);
      }
    } // mpi rank 0

  // Restart from diagonalization module
  case 2:
    /************************************************************/
    /*******************    RUN DIAG MODULE   *******************/
    /************************************************************/

    mod_diag(
        psitot, pot_local, eig_vals, sigma_E, &grid, G_vecs, k_vecs, LS, nlc, nl,
        an, zn, ene_targets, ksqr, &ist, &par, &flag, &parallel);

    // Save checkpoint if requested
    if (3 == flag.saveCheckpoints)
    {
      save_job_state(
          "checkpoint_3.dat", par.checkpoint_id, psitot, pot_local, ksqr,
          an, zn, ene_targets, nl, nlc, &grid, &ist, &par, &flag, &parallel);
    }

  case 3:
    /************************************************************/
    /*******************   RUN SIGMA MODULE   *******************/
    /************************************************************/
    /*** The periodic path computes the eigenvalue variance    ***/
    /*** inside mod_diag (calc_sigma_E_k), so the standalone   ***/
    /*** mod_sigma module is skipped when flag.periodic is set ***/

    if ((0 == mpir) && (0 == flag.periodic))
    {
      if (1 == flag.restartFromSigma)
      {
        restart_from_sigma(&psitot, pot_local, eig_vals, &grid, LS, nlc, nl,
                           ksqr, &ist, &par, &flag, &parallel);
      }
      printf("Entering mod_sigma\n");
      fflush(0);

      mod_sigma(psitot, pot_local, eig_vals, sigma_E, &grid, LS, nlc, nl,
                ksqr, &ist, &par, &flag, &parallel);
    }
  case 4:
    // Run post-processing -> mod_output or mod_optional output
    if (mpir == 0)
    {
      write_separation(stdout, "T");
      printf("\nPOST-PROCESSING | %s\n", get_time());
      write_separation(stdout, "B");
      fflush(stdout);

      if (4 == flag.restartFromCheckpoint)
      {
        // Initialize job in post-processing mode
        load_eval(eig_vals, sigma_E, &ist, &par);
        psitot = (double *)calloc(ist.mn_states_tot * ist.complex_idx * ist.nspinngrid, sizeof(double));
        load_psitot(psitot, &ist, &par);
      }
      /************************************************************/
      /*******************   RUN OUTPUT MODULE  *******************/
      /************************************************************/
      else
      {
        mod_output(psitot, R, eig_vals, sigma_E, &grid, &ist, &par, &flag, &parallel);
      }
    }
  } // end switch
  } // end else (non-spin-polarized path)

  /************************************************************/
  /*****************  OPTIONAL OUTPUT MODULE  *****************/
  /************************************************************/
  /*** Skipped for spin-polarized runs: run_spinpol already calls   ***/
  /*** mod_optional_output per spin channel (and frees psitot).     ***/

  if ((0 == mpir) && (0 == flag.periodic) && (0 == flag.spinPolarized))
  {
    mod_optional_output(psitot, &grid, &ist, &par, &flag, &parallel);
  }

  /************************************************************/
  /******************   FREE MEM & FINALIZE   *****************/
  /************************************************************/

  free(ist.atom_types);
  free(grid.x);
  free(grid.y);
  free(grid.z);
  free(R);
  free(atom);
  free(psi);
  free(phi);
  free(pot_local);
  free(ksqr);
  free(nlc);
  free(nl);
  free(eig_vals);
  free(ene_targets);
  free(sigma_E);
  free(an);
  free(zn);

  // Release the per-k-group communicator and mapping (periodic path)
  if (1 == flag.periodic)
  {
    if (parallel.k_comm != MPI_COMM_NULL)
      MPI_Comm_free(&parallel.k_comm);
    free(parallel.k_local_to_global);
  }

  if (0 == mpir)
  {
    free(psitot);

    // Get total job duration

    time_t end_time = time(NULL);
    time_t end_clock = clock();
    double elapsed_seconds = difftime(end_time, start_time);
    char *duration = format_duration(elapsed_seconds);

    // Print final job stat line

    write_separation(stdout, top);

    printf("\nDONE WITH PROGRAM: FILTER DIAGONALIZATION\n");
    printf("This calculation ended at: %s\n", ctime(&end_time));
    printf("Total job CPU time (sec) %.4g | %s\n",
           ((double)end_clock - (double)start_clock) / (double)(CLOCKS_PER_SEC),
           format_duration(((double)end_clock - (double)start_clock) / (double)(CLOCKS_PER_SEC)));
    printf("Total wall run time (sec) %.4g | %s", (double)end_time - (double)start_time, duration);

    write_separation(stdout, bottom);
    fflush(0);

    free(top);
    free(bottom);
  }

  MPI_Finalize(); // Finalize the MPI tasks and prepare to exit program

  exit(0);
} // End of main

/*****************************************************************************/
