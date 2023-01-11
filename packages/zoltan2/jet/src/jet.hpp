#include "contract.hpp"
#include "uncoarsen.hpp"
#include "initial_partition.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class partitioner {
public:

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
    using part_vt = Kokkos::View<part_t*, Device>;
    using coarsener_t = contracter<matrix_t>;
    using init_t = initial_partitioner<matrix_t, part_t>;
    using uncoarsener_t = uncoarsener<matrix_t, part_t>;
    using coarse_level_triple = typename coarsener_t::coarse_level_triple;

static part_vt partition(matrix_t g, wgt_view_t vweights, const part_t k, const double imb_ratio, bool uniform_ew,
                                  ExperimentLoggerUtil<scalar_t>& experiment) {

    coarsener_t coarsener;

    std::list<coarse_level_triple> cg_list;
    Kokkos::Timer t;
    double start_time = t.seconds();

    //coarsener.set_heuristic(coarsener_t::HECv1);
    coarsener.set_heuristic(coarsener_t::MtMetis);
    int cutoff = k*8;
    if(cutoff > 1024){
        cutoff = k*2;
        cutoff = std::max(1024, cutoff);
    }
    coarsener.set_coarse_vtx_cutoff(cutoff);
    coarsener.set_min_allowed_vtx(cutoff / 4);
    cg_list = coarsener.generate_coarse_graphs(g, vweights, experiment, uniform_ew);
    Kokkos::fence();
    double fin_coarsening_time = t.seconds();
    experiment.addMeasurement(Measurement::Coarsen, fin_coarsening_time - start_time);
    part_vt coarsest_p = init_t::metis_init(cg_list.back().mtx, cg_list.back().vtx_w, k, imb_ratio);
    //part_vt coarsest_p = init_t::random_init(cg_list.back().vtx_w, k, imb_ratio);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::InitPartition, t.seconds() - fin_coarsening_time);
    scalar_t edge_cut = 0;
    part_vt part = uncoarsener_t::uncoarsen(cg_list, coarsest_p, k, imb_ratio
        , edge_cut, experiment);

    Kokkos::fence();
    cg_list.clear();
    Kokkos::fence();
    double fin_time = t.seconds();
    experiment.addMeasurement(Measurement::Total, fin_time - start_time);

    return part;
}
};

}
