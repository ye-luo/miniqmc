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
#ifndef QMCPLUSPLUS_DTDIMPL_BA_H
#define QMCPLUSPLUS_DTDIMPL_BA_H

#include "OpenMP/OMPallocator.hpp"

namespace qmcplusplus
{
/**@ingroup nnlist
 * @brief A derived classe from DistacneTableData, specialized for AB using a
 * transposed form
 */
template<typename T, unsigned D, int SC>
struct DistanceTableBA : public DTD_BConds<T, D, SC>, public DistanceTableData
{
  int Nsources;
  int Ntargets;

  /// actual memory for Distances and Displacements
  Vector<RealType, OMPallocator<RealType, aligned_allocator<RealType>>> memoryPool;

  DistanceTableBA(const ParticleSet& source, ParticleSet& target)
      : DTD_BConds<T, D, SC>(source.Lattice), DistanceTableData(source, target)
  {
    resize(source.getTotalNum(), target.getTotalNum());
    #pragma omp target enter data map(to:this[:1])
  }

  void resize(int ns, int nt)
  {
    N[SourceIndex] = Nsources = ns;
    N[VisitorIndex] = Ntargets = nt;
    if (Nsources * Ntargets == 0)
      return;

    int Nsources_padded = getAlignedSize<T>(Nsources);

    memoryPool.resize(Ntargets * Nsources_padded * (D + 1));

    Distances.attachReference(memoryPool.data(), Ntargets, Nsources_padded);
    size_t head_offset = Ntargets * Nsources_padded;
    Displacements.resize(Ntargets);
    for (int i = 0; i < Ntargets; ++i)
      Displacements[i].attachReference(Nsources, Nsources_padded, memoryPool.data() + head_offset + i * Nsources_padded * D);

    Temp_r.resize(Nsources);
    Temp_dr.resize(Nsources);
  }

  DistanceTableBA()                       = delete;
  DistanceTableBA(const DistanceTableBA&) = delete;
  ~DistanceTableBA()
  {
    #pragma omp target exit data map(delete:this[:1])
  }

  /** evaluate the full table */
  inline void evaluate(ParticleSet& P)
  {
    // be aware of the sign of Displacement
    int Ntargets_local = Ntargets;
    int Ntargets_padded = getAlignedSize<T>(Ntargets);
    int Nsources_local = Nsources;
    int Nsources_padded = getAlignedSize<T>(Nsources);

    auto* target_pos_ptr = P.RSoA.data();
    const auto* source_pos_ptr = Origin->RSoA.data();
    auto* r_dr_ptr = memoryPool.data();

    const int ChunkSizePerTeam = 512;
    const int NumTeams         = (Nsources + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

    #pragma omp target teams distribute collapse(2) num_teams(Ntargets*NumTeams) \
      map(to: source_pos_ptr[:Nsources_padded*D], target_pos_ptr[:Ntargets_padded*D]) \
      map(always, from: r_dr_ptr[:memoryPool.size()]) nowait
    for (int iat = 0; iat < Ntargets; ++iat)
      for (int team_id = 0; team_id < NumTeams; team_id++)
      {
        const int first = ChunkSizePerTeam * team_id;
        const int last  = (first + ChunkSizePerTeam) > Nsources_local? Nsources_local : first + ChunkSizePerTeam;

        T pos[D];
        for(int idim = 0; idim<D; idim++)
          pos[idim] = target_pos_ptr[idim*Ntargets_padded + iat];

        auto* r_iat_ptr = r_dr_ptr + Nsources_padded * iat;
        auto* dr_iat_ptr = r_dr_ptr + Nsources_padded * Ntargets + Nsources_padded * D * iat;

        DTD_BConds<T, D, SC>::computeDistancesOffload(pos,
                                                      source_pos_ptr,
                                                      r_iat_ptr,
                                                      dr_iat_ptr,
                                                      Nsources_padded,
                                                      first,
                                                      last);
      }
  }

  /** evaluate the iat-row with the current position
   *
   * Fill Temp_r and Temp_dr and copy them Distances & Displacements
   */
  inline void evaluate(ParticleSet& P, IndexType iat)
  {
    DTD_BConds<T, D, SC>::computeDistances(P.R[iat],
                                           Origin->RSoA,
                                           Distances[iat],
                                           Displacements[iat],
                                           0,
                                           Nsources);
  }

  /// evaluate the temporary pair relations
  inline void move(const ParticleSet& P, const PosType& rnew)
  {
    DTD_BConds<T, D, SC>::computeDistances(rnew, Origin->RSoA, Temp_r.data(), Temp_dr, 0, Nsources);
  }

  /// update the stripe for jat-th particle
  inline void update(IndexType iat)
  {
    std::copy_n(Temp_r.data(), Nsources, Distances[iat]);
    for (int idim = 0; idim < D; ++idim)
      std::copy_n(Temp_dr.data(idim), Nsources, Displacements[iat].data(idim));
  }
};
} // namespace qmcplusplus
#endif
