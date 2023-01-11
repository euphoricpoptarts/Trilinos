// @HEADER
//
// ***********************************************************************
//
//   Zoltan2: A package of combinatorial algorithms for scientific computing
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Karen Devine      (kddevin@sandia.gov)
//                    Erik Boman        (egboman@sandia.gov)
//                    Siva Rajamanickam (srajama@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef _ZOLTAN2_ALGJET_HPP_
#define _ZOLTAN2_ALGJET_HPP_

#include <Zoltan2_GraphModel.hpp>
#include <Zoltan2_Algorithm.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_Util.hpp>
#include <Zoltan2_TPLTraits.hpp>
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <jet.hpp>

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


namespace Zoltan2 {

template <typename Adapter>
class AlgJet : public Algorithm<Adapter>
{
public:
  typedef typename Adapter::base_adapter_t base_adapter_t;
  typedef typename Adapter::lno_t lno_t;
  typedef typename Adapter::gno_t gno_t;
  typedef typename Adapter::offset_t offset_t;
  typedef typename Adapter::scalar_t scalar_t;
  typedef typename Adapter::part_t part_t;
  typedef typename Adapter::user_t user_t;
  typedef typename Adapter::userCoord_t userCoord_t;

  /*! PuLP constructors
   *  \param env          parameters for the problem and library configuration
   *  \param problemComm  the communicator for the problem
   *  \param adapter      the user's input adapter
   * 
   *  We're building a graph model, so throw an error if we can't  
   *    build the model from the input adapter passed to constructor
   *  For matrix and mesh adapters, additionally determine which 
   *    objects we wish to partition
   */
  AlgJet(const RCP<const Environment> &env__,
          const RCP<const Comm<int> > &problemComm__,
          const RCP<const IdentifierAdapter<user_t> > &adapter__) :
    env(env__), problemComm(problemComm__), adapter(adapter__)
  { 
    std::string errStr = "cannot build GraphModel from IdentifierAdapter, ";
    errStr            += "Jet requires Graph, Matrix, or Mesh Adapter";
    throw std::runtime_error(errStr);
  }  

  AlgJet(const RCP<const Environment> &env__,
          const RCP<const Comm<int> > &problemComm__,
          const RCP<const VectorAdapter<user_t> > &adapter__) :
    env(env__), problemComm(problemComm__), adapter(adapter__)
  { 
    std::string errStr = "cannot build GraphModel from VectorAdapter, ";
    errStr            += "Jet requires Graph, Matrix, or Mesh Adapter";
    throw std::runtime_error(errStr);
  }   

  AlgJet(const RCP<const Environment> &env__,
          const RCP<const Comm<int> > &problemComm__,
          const RCP<const GraphAdapter<user_t,userCoord_t> > &adapter__) :
    env(env__), problemComm(problemComm__), adapter(adapter__)
  { 
    modelFlag_t flags;
    flags.reset();

    buildModel(flags);
  }  

  AlgJet(const RCP<const Environment> &env__,
          const RCP<const Comm<int> > &problemComm__,
          const RCP<const MatrixAdapter<user_t,userCoord_t> > &adapter__) :
    env(env__), problemComm(problemComm__), adapter(adapter__)
  {   
    modelFlag_t flags;
    flags.reset();

    const ParameterList &pl = env->getParameters();
    const Teuchos::ParameterEntry *pe;

    std::string defString("default");
    std::string objectOfInterest(defString);
    pe = pl.getEntryPtr("objects_to_partition");
    if (pe)
      objectOfInterest = pe->getValue<std::string>(&objectOfInterest);

    if (objectOfInterest == defString ||
        objectOfInterest == std::string("matrix_rows") )
      flags.set(VERTICES_ARE_MATRIX_ROWS);
    else if (objectOfInterest == std::string("matrix_columns"))
      flags.set(VERTICES_ARE_MATRIX_COLUMNS);
    else if (objectOfInterest == std::string("matrix_nonzeros"))
      flags.set(VERTICES_ARE_MATRIX_NONZEROS);

    buildModel(flags);
  }

  AlgJet(const RCP<const Environment> &env__,
          const RCP<const Comm<int> > &problemComm__,
          const RCP<const MeshAdapter<user_t> > &adapter__) :
    env(env__), problemComm(problemComm__), adapter(adapter__)
  { 
    modelFlag_t flags;
    flags.reset();

    const ParameterList &pl = env->getParameters();
    const Teuchos::ParameterEntry *pe;

    std::string defString("default");
    std::string objectOfInterest(defString);
    pe = pl.getEntryPtr("objects_to_partition");
    if (pe)
      objectOfInterest = pe->getValue<std::string>(&objectOfInterest);

    if (objectOfInterest == defString ||
        objectOfInterest == std::string("mesh_nodes") )
      flags.set(VERTICES_ARE_MESH_NODES);
    else if (objectOfInterest == std::string("mesh_elements"))
      flags.set(VERTICES_ARE_MESH_ELEMENTS);

    buildModel(flags);
  }

  /*! \brief Set up validators specific to this algorithm
  */
  static void getValidParameters(ParameterList & pl)
  {
    pl.set("pulp_vert_imbalance", 1.03, "vertex imbalance tolerance, ratio of "
      "maximum load over average load",
      Environment::getAnyDoubleValidator());

    // bool parameter
    pl.set("pulp_do_repart", false, "perform repartitioning",
      Environment::getBoolValidator() );

    pl.set("pulp_seed", 0, "set pulp seed", Environment::getAnyIntValidator());
  }

  void partition(const RCP<PartitioningSolution<Adapter> > &solution);

private:

  void buildModel(modelFlag_t &flags);

  const RCP<const Environment> env;
  const RCP<const Comm<int> > problemComm;
  const RCP<const base_adapter_t> adapter;
  RCP<const GraphModel<base_adapter_t> > model;
};


/////////////////////////////////////////////////////////////////////////////
template <typename Adapter>
void AlgJet<Adapter>::buildModel(modelFlag_t &flags)
{   
  const ParameterList &pl = env->getParameters();
  const Teuchos::ParameterEntry *pe;

  std::string defString("default");
  std::string symParameter(defString);
  pe = pl.getEntryPtr("symmetrize_graph");
  if (pe){
    symParameter = pe->getValue<std::string>(&symParameter);
    if (symParameter == std::string("transpose"))
      flags.set(SYMMETRIZE_INPUT_TRANSPOSE);
    else if (symParameter == std::string("bipartite"))
      flags.set(SYMMETRIZE_INPUT_BIPARTITE);  } 

  bool sgParameter = false;
  pe = pl.getEntryPtr("subset_graph");
  if (pe)
    sgParameter = pe->getValue(&sgParameter);
  if (sgParameter)
      flags.set(BUILD_SUBSET_GRAPH);

  flags.set(REMOVE_SELF_EDGES);
  flags.set(GENERATE_CONSECUTIVE_IDS);
  flags.set(BUILD_LOCAL_GRAPH);
  this->env->debug(DETAILED_STATUS, "    building graph model");
  this->model = rcp(new GraphModel<base_adapter_t>(this->adapter, this->env, 
                                            this->problemComm, flags));
  this->env->debug(DETAILED_STATUS, "    graph model built");
}

template <typename Adapter>
void AlgJet<Adapter>::partition(
  const RCP<PartitioningSolution<Adapter> > &solution
)
{
  using Device = Kokkos::DefaultExecutionSpace;
  using matrix_t = KokkosSparse::CrsMatrix<scalar_t, gno_t, Device, void, offset_t>;
  using edge_vt = Kokkos::View<offset_t*, Device>;
  using edge_mt = Kokkos::View<const offset_t*, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using vtx_vt = Kokkos::View<gno_t*, Device>;
  using vtx_mt = Kokkos::View<const gno_t*, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using wgt_vt = Kokkos::View<scalar_t*, Device>;
  using wgt_mt = Kokkos::View<scalar_t*, Kokkos::DefaultHostExecutionSpace>;
  using part_vt = Kokkos::View<part_t*, Device>;
  using part_mt = Kokkos::View<part_t*, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  HELLO;

  size_t numGlobalParts = solution->getTargetGlobalNumberOfParts();

  int num_parts = (int)numGlobalParts;

  int np = problemComm->getSize();

  // Get number of vertices and edges
  const size_t modelVerts = model->getLocalNumVertices();
  const size_t modelEdges = model->getLocalNumEdges();
  gno_t num_verts = (gno_t)modelVerts;
  offset_t num_edges = (offset_t)modelEdges;
  //TPL_Traits<int, size_t>::ASSIGN(num_verts, modelVerts, env);
  //TPL_Traits<long, size_t>::ASSIGN(num_edges, modelEdges, env);

  // Get vertex info
  ArrayView<const gno_t> vtxIDs;
  ArrayView<StridedData<lno_t, scalar_t> > vwgts;
  size_t nVtx = model->getVertexList(vtxIDs, vwgts);
  int nVwgts = model->getNumWeightsPerVertex();


  // Jet currently only supports a single vertex weight
  if (nVwgts > 1) {
    std::cerr << "Warning:  NumWeightsPerVertex is " << nVwgts 
              << " but Jet allows only one weight. "
              << " Zoltan2 will use only the first weight per vertex."
              << std::endl;
  }

  wgt_mt vw_h(Kokkos::ViewAllocateWithoutInitializing("vertex weights view"), num_verts);
  scalar_t vertex_weights_sum = 0;
  if (nVwgts >= 1) {
    nVwgts = 1;
    for (gno_t i = 0; i < num_verts; ++i) {
      vw_h(i) = vwgts[0][i];
      vertex_weights_sum += vw_h(i);
    }
  } else {
      nVwgts = 1;
      Kokkos::deep_copy(vw_h, 1);
  }

  // Get edge info
  ArrayView<const gno_t> adjs;
  ArrayView<const offset_t> offsets;
  ArrayView<StridedData<lno_t, scalar_t> > ewgts;
  size_t nEdge = model->getEdgeList(adjs, offsets, ewgts);
  int nEwgts = model->getNumWeightsPerEdge();
  if (nEwgts > 1) {
    std::cerr << "Warning:  NumWeightsPerEdge is " << nEwgts 
              << " but Jet allows only one weight. "
              << " Zoltan2 will use only the first weight per edge."
              << std::endl;
  }

  bool uniform_ew = false;
  wgt_mt ew_h(Kokkos::ViewAllocateWithoutInitializing("edge weights view"), num_edges);
  if (nEwgts >= 1) {
    nEwgts = 1;
    for (offset_t i = 0; i < num_edges; i++){
      ew_h(i) = ewgts[0][i];
    }
  } else {
    uniform_ew = true;
    Kokkos::deep_copy(ew_h, 1);
  }

  vtx_mt adj_h(adjs.data(), num_edges);
  edge_mt off_h(offsets.data(), num_verts + 1);
  vtx_vt adj_d("edge adjacencies", num_edges);
  edge_vt off_d("row map", num_verts + 1);
  wgt_vt ew_d("edge weights", num_edges);
  wgt_vt vw_d("vertex weights", num_verts);
  Kokkos::deep_copy(adj_d, adj_h);
  Kokkos::deep_copy(off_d, off_h);
  Kokkos::deep_copy(ew_d, ew_h);
  Kokkos::deep_copy(vw_d, vw_h);
  matrix_t g("problem matrix", nVtx, nVtx, nEdge, ew_d, off_d, adj_d);

  // Create array to return results in.
  // Or write directly into solution parts array
  ArrayRCP<part_t> partList(new part_t[num_verts], 0, num_verts, true);
  part_mt partList_view(partList.get(), num_verts);

  // Grab options from parameter list
  const Teuchos::ParameterList &pl = env->getParameters();
  const Teuchos::ParameterEntry *pe;

  bool verbose_output = false;

  // Now grab vertex and edge imbalances, defaults at 3%
  double vert_imbalance = 1.03;
  double imbalance = 1.03;

  pe = pl.getEntryPtr("jet_vert_imbalance");
  if (pe) vert_imbalance = pe->getValue<double>(&vert_imbalance);
  pe = pl.getEntryPtr("jet_imbalance");
  if (pe) imbalance = pe->getValue<double>(&imbalance);

  if (vert_imbalance < 1.0)
    throw std::runtime_error("jet_vert_imbalance must be '1.0' or greater.");
  if (imbalance < 1.0)
    throw std::runtime_error("jet_imbalance must be '1.0' or greater.");

  pe = pl.getEntryPtr("jet_verbose");
  if (pe) verbose_output = pe->getValue(&verbose_output);

  // Call partitioning; result returned in parts array
  // Create Jet's partitioning data structure

  jet_partitioner::ExperimentLoggerUtil<scalar_t> experiment;
  part_vt output_part = jet_partitioner::partitioner<matrix_t, part_t>::partition(g, vw_d, num_parts, imbalance, uniform_ew, experiment);  

  if (verbose_output) {
      experiment.verboseReport();
  }

  // Load answer into the solution
  Kokkos::deep_copy(partList_view, output_part);

  solution->setParts(partList);

  env->memory("Zoltan2-Jet: After creating solution");

//#endif // DO NOT HAVE_MPI
}

} // namespace Zoltan2

////////////////////////////////////////////////////////////////////////


#endif

