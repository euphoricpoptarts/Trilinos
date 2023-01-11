#pragma once

#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

    /**********************************************************
     * Internal
     **********************************************************
     */
#if defined(SGPAR_HUGEGRAPHS)
typedef int64_t ordinal_t;
typedef int64_t edge_offset_t;
#elif defined(SGPAR_LARGEGRAPHS)
typedef int32_t ordinal_t;
typedef int64_t edge_offset_t;
#else
typedef int32_t ordinal_t;
typedef int32_t edge_offset_t;
#endif
typedef edge_offset_t value_t;
struct config_t {
    int coarsening_alg;
    int num_iter;
    double max_imb_ratio;
    int num_parts;
};

#if defined(SERIAL)
using Device = Kokkos::Serial;
#elif defined(HOST)
using Device = Kokkos::DefaultHostExecutionSpace;
#else
using Device = Kokkos::DefaultExecutionSpace;
#endif
using matrix_t = typename KokkosSparse::CrsMatrix<value_t, ordinal_t, Device, void, edge_offset_t>;
using host_matrix_t = typename KokkosSparse::CrsMatrix<value_t, ordinal_t, Kokkos::DefaultHostExecutionSpace, void, edge_offset_t>;
using graph_t = typename matrix_t::staticcrsgraph_type;
using host_graph_t = typename host_matrix_t::staticcrsgraph_type;

using host_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;

using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
using edge_mirror_t = typename edge_view_t::HostMirror;
using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
using gain_t = int32_t;
using gain_vt = Kokkos::View<gain_t*, Device>;
using gain_mt = typename gain_vt::HostMirror;
using vtx_t = Kokkos::View<ordinal_t, Device>;
using edge_t = Kokkos::View<edge_offset_t, Device>;
using vtx_mirror_t = typename vtx_view_t::HostMirror;
using wgt_view_t = Kokkos::View<value_t*, Device>;
using wgt_mirror_t = typename wgt_view_t::HostMirror;
using policy = Kokkos::TeamPolicy<typename Device::execution_space>;
using r_policy = Kokkos::RangePolicy<typename Device::execution_space>;
using member = typename policy::member_type;
using part_t = int;
using part_vt = Kokkos::View<part_t*, Device>;
using part_mt = typename part_vt::HostMirror;
