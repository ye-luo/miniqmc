////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_spo.cpp
 * @brief Miniapp to check 3D spline implementation against the reference.
 */
#include <Utilities/Configuration.h>
#include <Utilities/Communicate.h>
#include <Utilities/NewTimer.h>
#include <Drivers/Mover.hpp>
#include <Particle/ParticleSet.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Utilities/RandomGenerator.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/einspline_spo_omp.hpp>
#include <QMCWaveFunctions/einspline_spo_ref.hpp>
#include <Utilities/qmcpack_version.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

enum CheckSPOTimers
{
  Timer_Total,
  Timer_Init,
  Timer_SPO_v,
  Timer_SPO_vgh,
  Timer_SPO_ref_v,
  Timer_SPO_ref_vgh
};

TimerNameList_t<CheckSPOTimers> CheckSPOTimerNames = {
    {Timer_Total, "Total"},
    {Timer_Init, "Initialization"},
    {Timer_SPO_v, "SPO_v"},
    {Timer_SPO_vgh, "SPO_vgh"},
    {Timer_SPO_ref_v, "SPO_ref_v"},
    {Timer_SPO_ref_vgh, "SPO_ref_vgh"},
};

void print_help()
{
  // clang-format off
  app_summary() << "usage:" << '\n';
  app_summary() << "  check_spo [-fhvV] [-g \"n0 n1 n2\"] [-m meshfactor]"       << '\n';
  app_summary() << "            [-n steps] [-r rmax] [-s seed]"                  << '\n';
  app_summary() << "options:"                                                    << '\n';
  app_summary() << "  -g  set the 3D tiling.             default: 1 1 1"         << '\n';
  app_summary() << "  -f  results are not transfered back"                       << '\n';
  app_summary() << "  -h  print help and exit"                                   << '\n';
  app_summary() << "  -m  meshfactor                     default: 1.0"           << '\n';
  app_summary() << "  -n  number of MC steps             default: 5"             << '\n';
  app_summary() << "  -r  set the Rmax.                  default: 1.7"           << '\n';
  app_summary() << "  -s  set the random seed.           default: 11"            << '\n';
  app_summary() << "  -v  verbose output"                                        << '\n';
  app_summary() << "  -V  print version information and exit"                    << '\n';
  // clang-format on

  exit(1); // print help and exit
}

int main(int argc, char** argv)
{
  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  Communicate comm(argc, argv);

  // use the global generator

  int na      = 1;
  int nb      = 1;
  int nc      = 1;
  int nsteps  = 5;
  int nmovers = omp_get_max_threads();
  int iseed   = 11;
  // this is the cutoff from the non-local PP
  RealType Rmax(1.7);
  int nx = 37, ny = 37, nz = 37;
  int tileSize  = -1;
  bool transfer = true;
  bool verbose  = false;

  if (!comm.root())
  {
    outputManager.shutOff();
  }

  int opt;
  while (optind < argc)
  {
    if ((opt = getopt(argc, argv, "fhvVa:g:m:n:r:s:w:")) != -1)
    {
      switch (opt)
      {
      case 'a':
        tileSize = atoi(optarg);
        break;
      case 'g': // tiling1 tiling2 tiling3
        sscanf(optarg, "%d %d %d", &na, &nb, &nc);
        break;
      case 'f': // forward only, no transfer back to host
        printf("Results are not transferred back. Checking report should be ignored.\n");
        transfer = false;
        break;
      case 'h':
        print_help();
        break;
      case 'm':
      {
        const RealType meshfactor = atof(optarg);
        nx *= meshfactor;
        ny *= meshfactor;
        nz *= meshfactor;
      }
      break;
      case 'n': // number of MC steps
        nsteps = atoi(optarg);
        break;
      case 'r': // rmax
        Rmax = atof(optarg);
        break;
      case 's':
        iseed = atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'V':
        print_version(true);
        return 1;
        break;
      case 'w': // number of nmovers
        nmovers = atoi(optarg);
        break;
      default:
        print_help();
      }
    }
    else // disallow non-option arguments
    {
      app_error() << "Non-option arguments not allowed" << endl;
      print_help();
    }
  }

  if (comm.root())
  {
    if (verbose)
      outputManager.setVerbosity(Verbosity::HIGH);
    else
      outputManager.setVerbosity(Verbosity::LOW);
  }

  print_version(verbose);

  TimerManager.set_timer_threshold(timer_level_fine);
  TimerList_t Timers;
  setup_timers(Timers, CheckSPOTimerNames, timer_level_coarse);

  Timers[Timer_Total]->start();
  Timers[Timer_Init]->start();

  using spo_type = einspline_spo_omp<OHMMS_PRECISION>;
  spo_type spo_main;
  using spo_ref_type = miniqmcreference::einspline_spo_ref<OHMMS_PRECISION>;
  spo_ref_type spo_ref_main;
  int nTiles = 1;
  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  ParticleSet ions;
  // initialize ions and splines which are shared by all threads later
  {
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    build_ions(ions, tmat, lattice_b);
    const int norb = count_electrons(ions, 1) / 2;
    tileSize       = (tileSize > 0) ? tileSize : norb;
    nTiles         = norb / tileSize;

    const size_t SPO_coeff_size =
        static_cast<size_t>(norb) * (nx + 3) * (ny + 3) * (nz + 3) * sizeof(RealType);
    const double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;

    app_summary() << "Number of orbitals/splines = " << norb << endl
                  << "Tile size = " << tileSize << endl
                  << "Number of tiles = " << nTiles << endl
                  << "Rmax = " << Rmax << endl;
    app_summary() << "Iterations = " << nsteps << endl;
    app_summary() << "OpenMP threads = " << omp_get_max_threads() << endl;
#ifdef HAVE_MPI
    app_summary() << "MPI processes = " << comm.size() << endl;
#endif

    app_summary() << "\nSPO coefficients size = " << SPO_coeff_size << " bytes ("
                  << SPO_coeff_size_MB << " MB)" << endl;

    spo_main.set(nx, ny, nz, norb, nTiles);
    spo_main.Lattice.set(lattice_b);
    spo_ref_main.set(nx, ny, nz, norb, nTiles);
    spo_ref_main.Lattice.set(lattice_b);
  }

  const int nions = ions.getTotalNum();
  const int nels  = count_electrons(ions, 1);
  const int nels3 = 3 * nels;

  // construct a list of movers
  std::vector<Mover*> mover_list(nmovers, nullptr);
  std::cout << "Constructing " << nmovers << " movers!" << std::endl;
  std::vector<spo_type*> spo_views(nmovers, nullptr);
  std::vector<spo_ref_type*> spo_ref_views(nmovers, nullptr);
  // per mover data
  std::vector<ParticlePos_t> delta_list(nmovers);
  std::vector<ParticlePos_t> rOnSphere_list(nmovers);
  std::vector<std::vector<RealType>> ur_list(nmovers);
  std::vector<int> my_accepted_list(nmovers), my_vals_list(nmovers);
  std::vector<PosType> pos_list(nmovers);
  std::vector<SPOSet*> spo_shadows(nmovers);

#pragma omp parallel for
  for (size_t iw = 0; iw < mover_list.size(); iw++)
  {
    // create and initialize movers
    Mover* thiswalker = new Mover(MakeSeed(iw, mover_list.size()), ions);
    mover_list[iw]    = thiswalker;

    // create a spo view in each Mover
    spo_views[iw]     = new spo_type(spo_main, 1, 0);
    spo_shadows[iw]   = spo_views[iw];
    spo_ref_views[iw] = new spo_ref_type(spo_ref_main, 1, 0);

    // initial computing
    thiswalker->els.update();

    // temporal data during walking
    delta_list[iw].resize(nels);
    rOnSphere_list[iw].resize(thiswalker->nlpp.size());
    ur_list[iw].resize(nels);
  }

  Timers[Timer_Init]->stop();

  RealType sqrttau = 2.0;
  RealType accept  = 0.5;

  const double zval = 1.0 * static_cast<double>(nels) / static_cast<double>(nions);

  double ratio         = 0.0;
  double nspheremoves  = 0.0;
  double dNumVGHCalls  = 0.0;
  double evalV_v_err   = 0.0;
  double evalVGH_v_err = 0.0;
  double evalVGH_g_err = 0.0;
  double evalVGH_h_err = 0.0;

  for (int mc = 0; mc < nsteps; ++mc)
  {
    std::cout << "mc = " << mc << std::endl;
    for (size_t iw = 0; iw < mover_list.size(); iw++)
    {
      auto& mover     = *mover_list[iw];
      auto& random_th = mover.rng;
      auto& delta     = delta_list[iw];
      auto& ur        = ur_list[iw];

      random_th.generate_normal(&delta[0][0], nels3);
      random_th.generate_uniform(ur.data(), nels);
    }

    spo_type* anon_spo = spo_views[0];
    // VMC
    for (int iel = 0; iel < nels; ++iel)
    {
      for (size_t iw = 0; iw < mover_list.size(); iw++)
      {
        auto& els   = mover_list[iw]->els;
        auto& delta = delta_list[iw];
        auto& pos   = pos_list[iw];
        pos         = els.R[iel] + sqrttau * delta[iel];
      }

      Timers[Timer_SPO_vgh]->start();
      anon_spo->multi_evaluate_vgh(spo_shadows, pos_list);
      Timers[Timer_SPO_vgh]->stop();
      if (transfer)
        anon_spo->multi_transfer_vgh_from_device(spo_shadows);

#pragma omp parallel for reduction(+ : evalVGH_v_err, evalVGH_g_err, evalVGH_h_err)
      for (size_t iw = 0; iw < mover_list.size(); iw++)
      {
        auto& mover       = *mover_list[iw];
        auto& spo         = *spo_views[iw];
        auto& spo_ref     = *spo_ref_views[iw];
        auto& pos         = pos_list[iw];
        auto& els         = mover.els;
        auto& ur          = ur_list[iw];
        auto& my_accepted = my_accepted_list[iw];
        Timers[Timer_SPO_ref_vgh]->start();
        if (transfer)
          spo_ref.evaluate_vgh(pos);
        Timers[Timer_SPO_ref_vgh]->stop();
        // accumulate error
        for (int ib = 0; ib < spo.nBlocks; ib++)
          for (int n = 0; n < spo.nSplinesPerBlock; n++)
          {
            // value
            evalVGH_v_err += std::fabs(spo.psi[ib][n] - spo_ref.psi[ib][n]);
            // grad
            evalVGH_g_err += std::fabs(spo.grad[ib].data(0)[n] - spo_ref.grad[ib].data(0)[n]);
            evalVGH_g_err += std::fabs(spo.grad[ib].data(1)[n] - spo_ref.grad[ib].data(1)[n]);
            evalVGH_g_err += std::fabs(spo.grad[ib].data(2)[n] - spo_ref.grad[ib].data(2)[n]);
            // hess
            evalVGH_h_err += std::fabs(spo.hess[ib].data(0)[n] - spo_ref.hess[ib].data(0)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib].data(1)[n] - spo_ref.hess[ib].data(1)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib].data(2)[n] - spo_ref.hess[ib].data(2)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib].data(3)[n] - spo_ref.hess[ib].data(3)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib].data(4)[n] - spo_ref.hess[ib].data(4)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib].data(5)[n] - spo_ref.hess[ib].data(5)[n]);
          }
        if (ur[iel] > accept)
        {
          els.R[iel] = pos;
          my_accepted++;
        }
      }
    }

#if 0
#pragma omp parallel for reduction(+ : evalV_v_err)
        for(size_t iw = 0; iw < mover_list.size(); iw++)
      // TODO: move in mover loop
      auto &my_vals = my_vals_list[iw];
      my_vals = 0
      random_th.generate_uniform(ur.data(), nels);
      ecp.randomize(rOnSphere); // pick random sphere
      for (int iat = 0, kat = 0; iat < nions; ++iat)
      {
        const int nnF = static_cast<int>(ur[kat++] * zval);
        RealType r    = Rmax * ur[kat++];
        auto centerP  = ions.R[iat];
        my_vals += (nnF * ecp.size());

        for (int nn = 0; nn < nnF; ++nn)
        {
          for (int k = 0; k < ecp.size(); k++)
          {
            PosType pos = centerP + r * rOnSphere[k];
            spo.evaluate_v(pos);
            spo_ref.evaluate_v(pos);
            // accumulate error
            for (int ib = 0; ib < spo.nBlocks; ib++)
              for (int n = 0; n < spo.nSplinesPerBlock; n++)
                evalV_v_err += std::fabs(spo.psi[ib][n] - spo_ref.psi[ib][n]);
          }
        } // els
      } // ions
#endif

  } // steps.

  Timers[Timer_Total]->stop();

// free all movers
#pragma omp parallel for
  for (int iw = 0; iw < nmovers; iw++)
  {
    delete mover_list[iw];
    delete spo_views[iw];
    delete spo_ref_views[iw];
  }
  mover_list.clear();
  spo_views.clear();
  spo_ref_views.clear();

  outputManager.resume();

  if (comm.root())
  {
    cout << "================================== " << endl;
    TimerManager.print();
    cout << "================================== " << endl;
  }

  for (size_t iw = 0; iw < nmovers; iw++)
  {
    ratio += RealType(my_accepted_list[iw]) / RealType(nels * nsteps);
    nspheremoves += RealType(my_vals_list[iw]) / RealType(nsteps);
    dNumVGHCalls += nels;
  }

  //evalV_v_err /= nspheremoves;
  evalVGH_v_err /= dNumVGHCalls;
  evalVGH_g_err /= dNumVGHCalls;
  evalVGH_h_err /= dNumVGHCalls;

  const int np               = nmovers;
  constexpr RealType small_v = std::numeric_limits<RealType>::epsilon() * 1e4;
  constexpr RealType small_g = std::numeric_limits<RealType>::epsilon() * 3e6;
  constexpr RealType small_h = std::numeric_limits<RealType>::epsilon() * 6e8;
  int nfail                  = 0;
  app_log() << std::endl;
  if (evalV_v_err / np > small_v)
  {
    app_log() << "Fail in evaluate_v, V error =" << evalV_v_err / np << std::endl;
    nfail = 1;
  }
  if (evalVGH_v_err / np > small_v)
  {
    app_log() << "Fail in evaluate_vgh, V error =" << evalVGH_v_err / np << std::endl;
    nfail += 1;
  }
  if (evalVGH_g_err / np > small_g)
  {
    app_log() << "Fail in evaluate_vgh, G error =" << evalVGH_g_err / np << std::endl;
    nfail += 1;
  }
  if (evalVGH_h_err / np > small_h)
  {
    app_log() << "Fail in evaluate_vgh, H error =" << evalVGH_h_err / np << std::endl;
    nfail += 1;
  }
  comm.reduce(nfail);

  if (nfail == 0)
    app_log() << "All checks passed for spo" << std::endl;

  return 0;
}