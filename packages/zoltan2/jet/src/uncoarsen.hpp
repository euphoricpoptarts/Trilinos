#include "jet_refiner.hpp"
#include "contract.hpp"
#include <limits>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class uncoarsener {
public:
    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_mt = typename part_vt::HostMirror;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using coarsener_t = contracter<matrix_t>;
    using clt = typename coarsener_t::coarse_level_triple;
    using ref_t = jet_refiner<matrix_t, part_t>; 
    using rfd_t = typename ref_t::refine_data;
    using gain_t = typename ref_t::gain_t;
    using gain_vt = typename ref_t::gain_vt;

static double get_max_imb(gain_vt part_sizes, int k){
    typename gain_vt::HostMirror ps_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), part_sizes);
    gain_t imb = 0;
    gain_t fine_n = 0;
    for(int p = 0; p < k; p++){
        if(ps_host(p) > imb){
            imb = ps_host(p);
        }
        fine_n += ps_host(p);
    }
    return static_cast<double>(imb) / static_cast<double>(fine_n / k);
}

static void project(ordinal_t fine_n, vtx_view_t map, part_vt input, part_vt output){
    Kokkos::parallel_for("project", policy_t(0, fine_n), KOKKOS_LAMBDA(const ordinal_t i){
        output(i) = input(map(i));
    });
}

static part_vt multilevel_jet(std::list<clt> cg_list, part_vt coarse_guess, int k, const double imb_ratio, rfd_t& rfd, ExperimentLoggerUtil<scalar_t>& experiment, Kokkos::Timer& t){
    ref_t refiner(cg_list.front().mtx, k);

    clt cg = cg_list.back();
    cg_list.pop_back();
    while (!cg_list.empty()) {
        refiner.jet_refine(cg.mtx, k, imb_ratio, cg.vtx_w, coarse_guess, cg_list.size(), rfd, experiment);

        clt next_cg = cg_list.back();
        //interpolate
        part_vt fine_vec("fine vec", next_cg.mtx.numRows());
        project(next_cg.mtx.numRows(), cg.interp_mtx.map, coarse_guess, fine_vec);
        coarse_guess = fine_vec;
        cg = next_cg;
        cg_list.pop_back();
    }

    refiner.jet_refine(cg.mtx, k, imb_ratio, cg.vtx_w, coarse_guess, cg_list.size(), rfd, experiment);
    return coarse_guess;
}

static part_vt uncoarsen(std::list<clt> cg_list, part_vt coarsest, int k, double imb_ratio
    , scalar_t& ec, ExperimentLoggerUtil<scalar_t>& experiment) {

    Kokkos::Timer t;
    rfd_t rfd;
    part_vt res = multilevel_jet(cg_list, coarsest, k, imb_ratio, rfd, experiment, t);
    Kokkos::fence();
    double rtime = t.seconds();
    t.reset();
    ec = rfd.cut / 2;
    gain_vt part_sizes = rfd.part_sizes;
    typename gain_vt::HostMirror ps_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), part_sizes);
    gain_t largest = 0;
    gain_t total = rfd.total_size;
    double opt = static_cast<double>(total) / static_cast<double>(k);
    gain_t smallest = total;
    for(int p = 0; p < k; p++){
        if(ps_host(p) > largest){
            largest = ps_host(p);
        }
        if(ps_host(p) < smallest){
            smallest = ps_host(p);
        }
    }
    experiment.addMeasurement(Measurement::Refine, rtime);
    experiment.setFinestImbRatio(static_cast<double>(largest) / opt);
    experiment.setFinestEdgeCut(ec);
    experiment.setLargestPartSize(largest);
    experiment.setSmallestPartSize(smallest);
    return res;
}
};

}
