// Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
// Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights
// in this software.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef KRINO_INCLUDE_AKRI_CDMESH_REFINEMENT_H_
#define KRINO_INCLUDE_AKRI_CDMESH_REFINEMENT_H_

#include <Akri_CDFEM_Snapper.hpp>
#include <Akri_CDFEM_Support.hpp>
#include <stk_mesh/base/BulkData.hpp>

namespace krino {

class InterfaceGeometry;

void
mark_possible_cut_elements_for_adaptivity(const stk::mesh::BulkData& mesh,
      const RefinementInterface & refinement,
      const InterfaceGeometry & interfaceGeometry,
      const CDFEM_Support & cdfem_support,
      const int numRefinements);

void
mark_interface_elements_for_adaptivity(const stk::mesh::BulkData& mesh,
      const RefinementInterface & refinement,
      const InterfaceGeometry & interfaceGeometry,
      const std::vector<InterfaceID> & active_interface_ids,
      const CDFEM_Snapper & snapper,
      const CDFEM_Support & cdfem_support,
      const FieldRef coords_field,
      const int num_refinements);

}

#endif /* KRINO_INCLUDE_AKRI_CDMESH_REFINEMENT_H_ */
