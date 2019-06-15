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
  int BlockSize;

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

    int Ntargets_padded = getAlignedSize<T>(Ntargets);
    int Nsources_padded = getAlignedSize<T>(Nsources);

    Distances.resize(Ntargets, Nsources_padded);

    BlockSize = Nsources_padded * D;
    memoryPool.resize(Ntargets * BlockSize);
    Displacements.resize(Ntargets);
    for (int i = 0; i < Ntargets; ++i)
      Displacements[i].attachReference(Nsources, Nsources_padded, memoryPool.data() + i * BlockSize);

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
    /*
    for (int iat = 0; iat < Ntargets; ++iat)
      DTD_BConds<T, D, SC>::computeDistances(P.R[iat],
                                             Origin->RSoA,
                                             Distances[iat],
                                             Displacements[iat],
                                             0,
                                             Nsources);
    */
    for (int iat = 0; iat < Ntargets; ++iat)
    {
      int Nsources_padded = getAlignedSize<T>(Nsources);
      int Nsources_local = Nsources;
      T x = P.R[iat][0];
      T y = P.R[iat][1];
      T z = P.R[iat][2];
      auto* source_pos_ptr = Origin->RSoA.data();
      auto* r_ptr = Distances[iat];
      auto* dr_ptr = Displacements[iat].data();
      #pragma omp target map(to: source_pos_ptr[:Nsources_padded*D]) map(from: r_ptr[:Nsources_padded], dr_ptr[:Nsources_padded*D])
      {
        T pos[3] = {x, y, z};
        DTD_BConds<T, D, SC>::computeDistancesOffload(pos,
                                                      source_pos_ptr,
                                                      r_ptr,
                                                      dr_ptr,
                                                      Nsources_padded,
                                                      0,
                                                      Nsources_local);
      }
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
