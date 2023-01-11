#pragma once
#include <type_traits>
#include <limits>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "ExperimentLoggerUtil.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class jet_refiner {
public:

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    // need some trickery because make_signed is undefined for floating point types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using vtx_svt = Kokkos::View<ordinal_t, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_svt = Kokkos::View<part_t, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr gain_t GAIN_MIN = std::numeric_limits<gain_t>::lowest();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    static constexpr part_t NULL_PART = -1;
    static constexpr part_t HASH_RECLAIM = -2;

    static const ordinal_t max_sections = 128;
    static const int max_buckets = 25;

//data that is preserved between levels in the multilevel scheme
struct refine_data {
    gain_vt part_sizes;
    scalar_t total_size = 0;
    gain_t cut = 0;
    gain_t total_imb = 0;
    bool init = false;
};

//vertex-part connectivity data
struct conn_data {
    gain_vt conn_vals;
    edge_view_t conn_offsets;
    vtx_view_t lock_bit;
    part_vt dest_cache;
    part_vt conn_entries;
    part_vt conn_table_sizes;
};

//this struct contains all the scratch memory used by the refinement iterations
struct scratch_mem {
    gain_vt gain1, gain2, gain_persistent, evict_start, evict_end;
    vtx_view_t vtx1, vtx2, vtx3, zeros1;
    part_vt dest_part, undersized;
    vtx_svt counter1;
    gain_svt cc_part1;
    part_svt total_undersized;

    scratch_mem(const ordinal_t n, const ordinal_t min_size, const part_t k) {
        gain1 = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain scratch 1"), max(n, min_size));
        gain2 = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain scratch 2"), n);
        gain_persistent = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
        evict_start = gain_vt("evict start", k);
        evict_end = gain_vt("evict end", k);
        undersized = part_vt("undersized parts", k);
        vtx1 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 1"), n);
        vtx2 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), max(n, min_size));
        vtx3 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 3"), max(n, min_size));
        dest_part = part_vt(Kokkos::ViewAllocateWithoutInitializing("destination scratch"), n);
        zeros1 = vtx_view_t("zeros 1", n);
        counter1 = vtx_svt("counter 1");
        total_undersized = part_svt("total undersized");
        cc_part1 = gain_svt("cut change part1");
    }
};

    scratch_mem perm_scratch;
    conn_data perm_cdata;

    edge_offset_t count_gain_size(const matrix_t largest, part_t k){
        edge_offset_t gain_size = 0;
        Kokkos::parallel_reduce("comp offsets", policy_t(0, largest.numRows()), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update){
            ordinal_t degree = largest.graph.row_map(i + 1) - largest.graph.row_map(i);
            if(degree > static_cast<ordinal_t>(k)) degree = k;
            update += degree;
        }, gain_size);
        return gain_size;
    }

    jet_refiner(const matrix_t largest, part_t k) :
        perm_scratch(largest.numRows(), k*max_sections*max_buckets, k) {
        ordinal_t n = largest.numRows();
        edge_view_t conn_offsets("gain offsets", n + 1);
        edge_offset_t gain_size = count_gain_size(largest, k);
        perm_cdata.conn_vals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("conn vals"), gain_size);
        perm_cdata.conn_entries = part_vt(Kokkos::ViewAllocateWithoutInitializing("conn entries"), gain_size);
        perm_cdata.conn_offsets = conn_offsets;
        perm_cdata.dest_cache = part_vt(Kokkos::ViewAllocateWithoutInitializing("best connected part for each vertex"), n);
        perm_cdata.conn_table_sizes = part_vt(Kokkos::ViewAllocateWithoutInitializing("map size"), n);
        perm_cdata.lock_bit = vtx_view_t("lock bit", n);
    }

gain_t get_total_cut(const matrix_t g, const part_vt partition){
    gain_t total_cut = 0;
    if(!is_host_space ){
        Kokkos::parallel_reduce("find total cut (team)", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t, gain_t& update){
            gain_t local_cut = 0;
            ordinal_t i = t.league_rank();
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j, gain_t& local_update){
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                if(partition(i) != partition(v)){
                    local_update += wgt;
                }
            }, local_cut);
            Kokkos::single(Kokkos::PerTeam(t), [&] (){
                update += local_cut;
            });
        }, total_cut);
    } else {
        Kokkos::parallel_reduce("find total cut", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
            gain_t local_cut = 0;
            for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                if(partition(i) != partition(v)){
                    local_cut += wgt;
                }
            }
            update += local_cut;
        }, total_cut);
    }
    return total_cut;
}

gain_vt get_part_sizes(const matrix_t g, const wgt_view_t vtx_w, const part_vt partition, part_t k){
    gain_vt part_size("part sizes", k);
    Kokkos::parallel_for("calc part sizes", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = partition(i);
        Kokkos::atomic_add(&part_size(p), vtx_w(i));
    });
    return part_size;
}

//get sum of vertex weights
scalar_t get_total_size(const matrix_t g, const wgt_view_t vtx_w){
    scalar_t total_size = 0;
    Kokkos::parallel_reduce("sum of vertex weights", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        update += vtx_w(i);
    }, total_size);
    return total_size;
}

//8 kernels, 2 device-host syncs
vtx_view_t jet_lp(const matrix_t& g, const part_t k, const part_vt& part, const conn_data& cdata, scratch_mem& scratch, bool top_level){
    ordinal_t n = g.numRows();
    ordinal_t num_pos = 0;
    vtx_view_t swap_scratch = scratch.vtx1;
    part_vt dest_part = scratch.dest_part;
    part_vt conn_entries = cdata.conn_entries;
    edge_view_t conn_offsets = cdata.conn_offsets;
    gain_vt conn_vals = cdata.conn_vals;
    gain_vt save_gains = scratch.gain_persistent;
    vtx_view_t lock_bit = cdata.lock_bit;
    //find a potential destination for each vertex among parts it is adjacent to
    //filter these potential moves
    //a vertex will only move if dest_part(i) != part(i) after this kernel
    Kokkos::parallel_for("select destination part (lp)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t best = cdata.dest_cache(i);
        if(best != NULL_PART) {
            dest_part(i) = best;
        } else {
            part_t p = part(i);
            gain_t gain = 0;
            edge_offset_t start = conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            best = p;
            for(edge_offset_t j = start; j < end; j++){
                if(conn_vals(j) > gain && conn_entries(j) != p){
                    best = conn_entries(j);
                    gain = conn_vals(j);
                }
            }
            save_gains(i) = 0;
            //gain > 0 if we found a part neighboring i that isn't part(i)
            if(gain > 0){
                //check connectivity to p
                gain_t p_gain = 0;
                if(size < k){
                    for(part_t q = 0; q < size; q++){
                        part_t p_i = (p + q) % size;
                        if(conn_entries(start + p_i) == p){
                            p_gain = conn_vals(start + p_i);
                            break;
                        } else if(conn_entries(start + p_i) == NULL_PART){
                            p_gain = 0;
                            break;
                        }
                    }
                } else {
                    p_gain = conn_vals(start + p);
                }
                //these conditions filter vertices that may potentially move
                //as a function of the gain of that move
                //and the connectivity to part(i)
                if(top_level){
                    if(gain >= p_gain || ((p_gain - gain) < floor(0.25*p_gain))){
                        save_gains(i) = gain - p_gain;
                    } else {
                        best = p;
                    }
                } else {
                    if(gain >= p_gain || ((p_gain - gain) < floor(0.75*p_gain))) {
                        save_gains(i) = gain - p_gain;
                    } else {
                        best = p;
                    }
                }
            }
            cdata.dest_cache(i) = best;
            dest_part(i) = best;
        }
    });
    //need to store the pre-afterburn gains into a separate view
    //than savegains, because we write new values into it that may not be overwritten
    //if a vertex has its best neighbor cached
    gain_vt pregain = scratch.gain1;
    //scan all unlocked vertices that passed the above filter
    //output count of such vertices into num_pos
    Kokkos::parallel_scan("filter potentially viable moves", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
            part_t p = part(i);
            part_t best = dest_part(i);
            if(p != best && lock_bit(i) == 0){
                if(final){
                    swap_scratch(update) = i;
                    pregain(i) = save_gains(i);
                }
                update++;
            } else if(final){
                pregain(i) = GAIN_MIN;
            }
    }, num_pos);
    //truncate scratch view by num_pos
    vtx_view_t pos_moves = Kokkos::subview(swap_scratch, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    vtx_view_t should_swap = Kokkos::subview(scratch.zeros1, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //in this kernel every potential move from the previous filters
    //is reevaluated by considering the effect of the other potential moves
    //a move is considered to occur before another according to their potential gains
    //and the vertex ids
    Kokkos::parallel_for("afterburner heuristic", team_policy_t(num_pos, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        gain_t change = 0;
        ordinal_t i = pos_moves(t.league_rank());
        part_t best = dest_part(i);
        part_t p = part(i);
        gain_t igain = pregain(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&](const edge_offset_t j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t vgain = pregain(v);
            if(vgain > igain || (vgain == igain && v < i)){
                part_t vpart = dest_part(v);
                scalar_t wgt = g.values(j);
                if(vpart == p){
                    update -= wgt;
                } else if(vpart == best){
                    update += wgt;
                }
                vpart = part(v);
                if(vpart == p){
                    update += wgt;
                } else if(vpart == best){
                    update -= wgt;
                }
            }
        }, change);
        t.team_barrier();
        Kokkos::single(Kokkos::PerTeam(t), [&](){
            if(igain + change >= 0){
                should_swap(t.league_rank()) = 1;
            }
        });
    });
    vtx_view_t swaps2 = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //scan all vertices that passed the post filter
    Kokkos::parallel_scan("filter beneficial moves", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
            if(should_swap(i)){
                if(final){
                    swaps2(update) = pos_moves(i);
                    //unset this because this memory is used for other things
                    should_swap(i) = 0;
                }
                update++;
            }
    }, num_pos);
    pos_moves = Kokkos::subview(swaps2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    Kokkos::deep_copy(exec_space(), lock_bit, 0);
    Kokkos::parallel_for("set lock bit", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t x){
        lock_bit(pos_moves(x)) = 1;
    });
    return pos_moves;
}

//at most 14 kernels, 2 device-host syncs
vtx_view_t rebalance_strong(const matrix_t& g, const part_t k, const wgt_view_t& vtx_w, const part_vt& part, const conn_data& cdata, const ordinal_t t_buckets, const gain_t opt_size, const double imb_ratio, scratch_mem& scratch, gain_vt part_sizes){
    ordinal_t n = g.numRows();
    //GPU version
    ordinal_t sections = max_sections;
    ordinal_t section_size = (n + sections*k) / (sections*k);
    if(section_size < 4096){
        section_size = 4096;
        sections = (n + section_size*k) / (section_size*k);
    }
    //use minibuckets within each gain bucket to reduce atomic contention
    //because the number of gain buckets is small
    ordinal_t t_minibuckets = t_buckets*k*sections;
    vtx_view_t bucket_sizes = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets));
    Kokkos::deep_copy(exec_space(), bucket_sizes, 0);
    //atomically count vertices in each gain bucket
    gain_t size_max = imb_ratio*opt_size;
    gain_t max_dest = opt_size + 1;
    Kokkos::parallel_for("assign move scores part1", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        if(part_sizes(p) > size_max){
            gain_t gain = 0;//= -gains(i*k + p);
            uint64_t tk = 0;
            uint64_t tg = 0;
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            gain_t p_gain = 0;
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                } else if(pj > NULL_PART) {
                    if(part_sizes(pj) < max_dest){
                        tg += cdata.conn_vals(j);
                        tk += 1;
                    }
                }
            }
            if(tk == 0) tk = 1;
            gain = (tg / tk) - p_gain;
            ordinal_t gain_type = t_buckets;
            //cast to int so we can approximate log_2
            int gx = gain;
            if(gx > 0){
                gain_type = 0;
            } else {
                gain_type = 1;
                gx = abs(gx);
                while(gx > 0){
                    gx >>= 1;
                    gain_type++;
                }
            }
            if(gain_type < t_buckets && vtx_w(i) < 2*(part_sizes(p) - opt_size)){
                ordinal_t g_id = (t_buckets*p + gain_type) * sections + (i % sections);
                Kokkos::atomic_increment(&bucket_sizes(g_id));
            }
        }
    });
    vtx_view_t bucket_offsets = Kokkos::subview(scratch.vtx3, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets + 1));
    //scan bucket sizes to compute offsets
    if(t_minibuckets < 10000 && !is_host_space){
        Kokkos::parallel_for("scan scores", team_policy_t(1, 1024), KOKKOS_LAMBDA(const member& t){
            //this scan is small so do it within a team instead of an entire grid to save kernel launch time
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets), [&] (const ordinal_t i, ordinal_t& update, const bool final) {
                update += bucket_sizes(i);
                if(final){
                    if(i == 0){
                        bucket_offsets(i) = 0;
                    }
                    bucket_offsets(i + 1) = update;
                    //reset gain counts here to save on a kernel launch
                    bucket_sizes(i) = 0;
                }
            });
        });
    } else {
        Kokkos::parallel_scan("scan scores", policy_t(0, t_minibuckets), KOKKOS_LAMBDA(const ordinal_t& i, ordinal_t& update, const bool final){
            update += bucket_sizes(i);
            if(final){
                if(i == 0){
                    bucket_offsets(i) = 0;
                }
                bucket_offsets(i + 1) = update;
                //reset gain counts here to save on a kernel launch
                bucket_sizes(i) = 0;
            }
        });
    }
    vtx_svt total_gain_s = Kokkos::subview(bucket_offsets, t_minibuckets);
    ordinal_t total_gain = 0;
    Kokkos::deep_copy(exec_space(), total_gain, total_gain_s);
    vtx_view_t least_bad_moves = Kokkos::subview(scratch.vtx1, std::make_pair(static_cast<ordinal_t>(0), total_gain));
    //atomically count vertices again, write vertices to appropriate location
    Kokkos::parallel_for("assign move scores part2", Kokkos::Experimental::require(policy_t(0, n), Kokkos::Experimental::WorkItemProperty::HintHeavyWeight), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        if(part_sizes(p) > size_max){
            gain_t gain = 0;//-gains(i*k + p);
            uint64_t tk = 0;
            uint64_t tg = 0;
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            gain_t p_gain = 0;
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                } else if(pj > NULL_PART) {
                    if(part_sizes(pj) < max_dest){
                        tg += cdata.conn_vals(j);
                        tk += 1;
                    }
                }
            }
            if(tk == 0) tk = 1;
            gain = (tg / tk) - p_gain;
            ordinal_t gain_type = t_buckets;
            //cast to int so we can approximate log_2
            int gx = gain;
            if(gx > 0){
                gain_type = 0;
            } else {
                gain_type = 1;
                gx = abs(gx);
                while(gx > 0){
                    gx >>= 1;
                    gain_type++;
                }
            }
            if(gain_type < t_buckets && vtx_w(i) < 2*(part_sizes(p) - opt_size)){
                ordinal_t g_id = (t_buckets*p + gain_type) * sections + (i % sections);
                ordinal_t insert = Kokkos::atomic_fetch_add(&bucket_sizes(g_id), 1);
                insert += bucket_offsets(g_id);
                least_bad_moves(insert) = i;
            }
        }
    });
    ordinal_t t_vtx = least_bad_moves.extent(0);
    gain_vt balance_scan = Kokkos::subview(scratch.gain1, std::make_pair(static_cast<ordinal_t>(0), t_vtx + 1));
    Kokkos::parallel_scan("assign move scores part3", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update, const bool final){
        ordinal_t x = least_bad_moves(i);
        update += vtx_w(x);
        if(final){
            balance_scan(i + 1) = update;
            if(i == 0){
                balance_scan(i) = 0;
            }
        }
    });
    gain_vt evict_start = scratch.evict_start;
    gain_vt evict_end = scratch.evict_end;
    Kokkos::parallel_for("find score cutoffs", policy_t(0, k), KOKKOS_LAMBDA(const int idx){
        evict_start(idx) = bucket_offsets(idx*t_buckets*sections);
        if(part_sizes(idx) > size_max){
            gain_t evict_total = part_sizes(idx) - size_max;
            ordinal_t start = bucket_offsets(idx*t_buckets*sections);
            ordinal_t end = bucket_offsets((idx + 1)*t_buckets*sections);
            gain_t find = balance_scan(start) + evict_total;
            ordinal_t mid = (start + end) / 2;
            //binary search to find eviction cutoffs for each k
            while(start + 1 < end){
                if(balance_scan(mid) >= find){
                    end = mid;
                } else {
                    start = mid;
                }
                mid = (start + end) / 2;
            }
            //if(abs(balance_scan(end) - find) < abs(balance_scan(start) - find)){
                evict_end(idx) = end;
            //} else {
            //    evict_end(idx) = start;
            //}
        } else {
            evict_end(idx) = bucket_offsets(idx*t_buckets*sections);
        }
    });
    vtx_view_t moves = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), total_gain));
    ordinal_t num_moves = 0;
    Kokkos::parallel_scan("filter below cutoffs", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t u = least_bad_moves(i);
        part_t p = part(u);
        if(i < evict_end(p)){
            if(final){
                moves(update) = u;
            }
            update++;
        }
    }, num_moves);
    t_vtx = num_moves;
    vtx_view_t only_moves = Kokkos::subview(moves, std::make_pair(static_cast<ordinal_t>(0), t_vtx));
    part_vt dest_part = scratch.dest_part;
    vtx_view_t unassigned = only_moves;
    Kokkos::parallel_scan("balance scan", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update, const bool final){
        ordinal_t x = unassigned(i);
        update += vtx_w(x);
        if(final){
            balance_scan(i + 1) = update;
            if(i == 0){
                balance_scan(i) = 0;
            }
        }
    });
    Kokkos::parallel_for("cookie cutter", policy_t(0, 1), KOKKOS_LAMBDA(const int idx){
        if(idx == 0){
            evict_start(0) = 0;
            for(int p = 0; p < k; p++){
                gain_t select = 0;
                if(max_dest > part_sizes(p)){
                    select = max_dest - part_sizes(p);
                }
                ordinal_t start = evict_start(p);
                ordinal_t end = t_vtx;
                gain_t find = balance_scan(start) + select;
                ordinal_t mid = (start + end) / 2;
                //binary search to find eviction cutoffs for each k
                while(start + 1 < end){
                    if(balance_scan(mid) >= find){
                        end = mid;
                    } else {
                        start = mid;
                    }
                    mid = (start + end) / 2;
                }
                if(abs(balance_scan(end) - find) < abs(balance_scan(start) - find)){
                    evict_end(p) = end;
                } else {
                    evict_end(p) = start;
                }
                if(p + 1 < k){
                    evict_start(p+1) = evict_end(p);
                }
            }
        }
    });
    Kokkos::parallel_for("select destination parts (rs)", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i){
        int p = 0;
        while(p < k && evict_start(p) <= i){
            p++;
        }
        p--;
        if(i < evict_end(p)){
            dest_part(unassigned(i)) = p;
        } else {
            dest_part(unassigned(i)) = part(unassigned(i));
        }
    });
    return only_moves;
}

//an unused version of rebalance_weak that fills all undersized parts instead of draining all oversized parts
vtx_view_t rebalance_pull(const matrix_t& g, const part_t k, const wgt_view_t& vtx_w, const part_vt& part, const conn_data& cdata, const ordinal_t t_buckets, const gain_t opt_size, const double imb_ratio, scratch_mem& scratch, gain_vt part_sizes){
    ordinal_t n = g.numRows();
    //GPU version
    ordinal_t sections = max_sections;
    ordinal_t section_size = (n + sections*k) / (sections*k);
    if(section_size < 4096){
        section_size = 4096;
        sections = (n + section_size*k) / (section_size*k);
    }
    //use minibuckets within each gain bucket to reduce atomic contention
    //because the number of gain buckets is small
    ordinal_t t_minibuckets = t_buckets*k*sections;
    gain_vt bucket_offsets = Kokkos::subview(scratch.gain1, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets));
    gain_vt bucket_sizes = bucket_offsets;
    Kokkos::deep_copy(exec_space(), bucket_sizes, 0);
    part_vt dest_part = scratch.dest_part;
    gain_t size_max = (2.0 - imb_ratio)*opt_size;
    gain_vt save_gains = scratch.gain2;
    part_vt undersized = scratch.undersized;
    part_svt total_undersized = scratch.total_undersized;
    Kokkos::parallel_for("init undersized parts list", team_policy_t(1, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        //this scan is small so do it within a team instead of an entire grid to save kernel launch time
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, k), [&] (const part_t i, part_t& update, const bool final) {
            if(part_sizes(i) < size_max){
                if(final){
                    undersized(update) = i;
                }
                update++;
            }
            if(final && i + 1 == k){
                total_undersized() = update;
            }
        });
    });
    Kokkos::parallel_for("select destination parts (rw)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
        part_t p = part(i);
        gain_t p_gain = 0;
        part_t best = p;
        gain_t gain = 0;
        if(part_sizes(p) > opt_size){
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj > NULL_PART && part_sizes(pj) < size_max){
                    gain_t jgain = cdata.conn_vals(j);
                    if(jgain > gain){
                            best = pj;
                            gain = jgain;
                    }
                }
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                }
            }
            if(gain > 0){
                dest_part(i) = best;
                save_gains(i) = gain - p_gain;
            } else {
                best = undersized(i % total_undersized());
                dest_part(i) = best;
                save_gains(i) = -p_gain;
            }
        } else {
            dest_part(i) = p;
        }
    });
    gain_vt vscore = save_gains;
    vtx_view_t bid = scratch.vtx3;
    //atomically count vertices in each gain bucket
    Kokkos::parallel_for("assign move scores", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        part_t best = dest_part(i);
        bid(i) = -1;
        if(p != best){
            //cast to int so we can approximate log_2
            int gain = save_gains(i);
            ordinal_t gain_type = t_buckets;
            if(gain > 0){
                gain_type = 0;
            } else {
                gain_type = 1;
                gain = abs(gain);
                while(gain > 0){
                    gain >>= 1;
                    gain_type++;
                }
            }
            if(gain_type < t_buckets){
                ordinal_t g_id = (t_buckets*best + gain_type) * sections + (i % sections);
                bid(i) = g_id;
                vscore(i) = Kokkos::atomic_fetch_add(&bucket_sizes(g_id), vtx_w(i));
            }
        }
    });
    //exclusive prefix sum to compute offsets
    //bucket_sizes is an alias of bucket_offsets
    if(t_minibuckets < 10000 && !is_host_space){
        Kokkos::parallel_for("scan score buckets", team_policy_t(1, 1024), KOKKOS_LAMBDA(const member& t){
            //this scan is small so do it within a team instead of an entire grid to save kernel launch time
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets), [&] (const ordinal_t i, gain_t& update, const bool final) {
                gain_t x = bucket_sizes(i);
                if(final){
                    bucket_offsets(i) = update;
                }
                update += x;
            });
        });
    } else {
        Kokkos::parallel_scan("scan score buckets", policy_t(0, t_minibuckets), KOKKOS_LAMBDA(const ordinal_t& i, gain_t& update, const bool final){
            gain_t x = bucket_sizes(i);
            if(final){
                bucket_offsets(i) = update;
            }
            update += x;
        });
    }
    vtx_view_t moves = scratch.vtx1;
    ordinal_t num_moves = 0;
    Kokkos::parallel_scan("filter scores below cutoff", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t b = bid(i);
        if(b != -1){
            part_t p = dest_part(i);
            ordinal_t begin_bucket = t_buckets*p*sections;
            gain_t score = vscore(i) + bucket_offsets(b) - bucket_offsets(begin_bucket);
            gain_t limit = size_max - part_sizes(p);
            if(score < limit){
                if(final){
                    moves(update) = i;
                }
                update++;
            }
        }
    }, num_moves);
    vtx_view_t only_moves = Kokkos::subview(moves, std::make_pair(static_cast<ordinal_t>(0), num_moves));
    return only_moves;
}

//at most 8 kernels, 1 device-host sync
vtx_view_t rebalance_weak(const matrix_t& g, const part_t k, const wgt_view_t& vtx_w, const part_vt& part, const conn_data& cdata, const ordinal_t t_buckets, const gain_t opt_size, const double imb_ratio, scratch_mem& scratch, gain_vt part_sizes){
    ordinal_t n = g.numRows();
    //GPU version
    ordinal_t sections = max_sections;
    ordinal_t section_size = (n + sections*k) / (sections*k);
    if(section_size < 4096){
        section_size = 4096;
        sections = (n + section_size*k) / (section_size*k);
    }
    //use minibuckets within each gain bucket to reduce atomic contention
    //because the number of gain buckets is small
    ordinal_t t_minibuckets = t_buckets*k*sections;
    gain_vt bucket_offsets = Kokkos::subview(scratch.gain1, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets + 1));
    gain_vt bucket_sizes = bucket_offsets;
    Kokkos::deep_copy(exec_space(), bucket_sizes, 0);
    part_vt dest_part = scratch.dest_part;
    gain_t size_max = imb_ratio*opt_size;
    gain_vt save_gains = scratch.gain2;
    part_vt undersized = scratch.undersized;
    gain_t max_dest = size_max*0.99;
    if(max_dest < size_max - 100){
        max_dest = size_max - 100;
    }
    part_svt total_undersized = scratch.total_undersized;
    Kokkos::parallel_for("init undersized parts list", team_policy_t(1, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        //this scan is small so do it within a team instead of an entire grid to save kernel launch time
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, k), [&] (const part_t i, part_t& update, const bool final) {
            if(part_sizes(i) < max_dest){
                if(final){
                    undersized(update) = i;
                }
                update++;
            }
            if(final && i + 1 == k){
                total_undersized() = update;
            }
        });
    });
    Kokkos::parallel_for("select destination parts (rw)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
        part_t p = part(i);
        gain_t p_gain = 0;
        part_t best = p;
        gain_t gain = 0;
        if(part_sizes(p) > size_max && vtx_w(i) < 1.5*(part_sizes(p) - opt_size)){
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj > NULL_PART && part_sizes(pj) < max_dest){
                    gain_t jgain = cdata.conn_vals(j);
                    if(jgain > gain){
                            best = pj;
                            gain = jgain;
                    }
                }
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                }
            }
            if(gain > 0){
                dest_part(i) = best;
                save_gains(i) = gain - p_gain;
            } else {
                best = undersized(i % total_undersized());
                dest_part(i) = best;
                save_gains(i) = -p_gain;
            }
        } else {
            dest_part(i) = p;
        }
    });
    gain_vt vscore = save_gains;
    vtx_view_t bid = scratch.vtx3;
    //atomically count vertices in each gain bucket
    Kokkos::parallel_for("assign move scores", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        part_t best = dest_part(i);
        bid(i) = -1;
        if(p != best){
            //cast to int so we can approximate log_2
            int gain = save_gains(i);
            ordinal_t gain_type = t_buckets;
            if(gain > 0){
                gain_type = 0;
            } else {
                gain_type = 1;
                gain = abs(gain);
                while(gain > 0){
                    gain >>= 1;
                    gain_type++;
                }
            }
            if(gain_type < t_buckets){
                ordinal_t g_id = (t_buckets*p + gain_type) * sections + (i % sections);
                bid(i) = g_id;
                vscore(i) = Kokkos::atomic_fetch_add(&bucket_sizes(g_id), vtx_w(i));
            }
        }
    });
    //exclusive prefix sum to compute offsets
    //bucket_sizes is an alias of bucket_offsets
    if(t_minibuckets < 10000 && !is_host_space){
        Kokkos::parallel_for("scan score buckets", team_policy_t(1, 1024), KOKKOS_LAMBDA(const member& t){
            //this scan is small so do it within a team instead of an entire grid to save kernel launch time
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets), [&] (const ordinal_t i, gain_t& update, const bool final) {
                gain_t x = bucket_sizes(i);
                if(final){
                    bucket_offsets(i) = update;
                }
                update += x;
            });
        });
    } else {
        Kokkos::parallel_scan("scan score buckets", policy_t(0, t_minibuckets), KOKKOS_LAMBDA(const ordinal_t& i, gain_t& update, const bool final){
            gain_t x = bucket_sizes(i);
            if(final){
                bucket_offsets(i) = update;
            }
            update += x;
        });
    }
    vtx_view_t moves = scratch.vtx1;
    ordinal_t num_moves = 0;
    Kokkos::parallel_scan("filter scores below cutoff", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t b = bid(i);
        if(b != -1){
            part_t p = part(i);
            ordinal_t begin_bucket = t_buckets*p*sections;
            gain_t score = vscore(i) + bucket_offsets(b) - bucket_offsets(begin_bucket);
            gain_t limit = part_sizes(p) - size_max;
            if(score < limit){
                if(final){
                    moves(update) = i;
                }
                update++;
            }
        }
    }, num_moves);
    vtx_view_t only_moves = Kokkos::subview(moves, std::make_pair(static_cast<ordinal_t>(0), num_moves));
    return only_moves;
}

//5 kernels, 1 device-host sync
void swap_and_update_alt(const matrix_t g, const part_t k, const wgt_view_t vtx_w, part_vt part, const vtx_view_t swaps, const part_vt dest_part, vtx_view_t swap_bit, conn_data cdata, gain_t& cut_change, gain_vt part_sizes, scratch_mem& scratch){
    ordinal_t total_moves = swaps.extent(0);
    //this reduction effectively takes place over two separate kernels
    //in order to avoid an extra copy to host, we store the first result on the device
    //and add it into the second result during the second kernel later
    //so that both can be transferred in a single copy to host
    gain_svt cc_part1 = scratch.cc_part1;
    Kokkos::parallel_reduce("count cutsize change part1", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        part_t p_con = 0;
        part_t b_con = 0;
        if(size < k){
            for(part_t q = 0; q < size; q++){
                part_t p_i = (best + q) % size;
                if(cdata.conn_entries(start + p_i) == best){
                    b_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
            for(part_t q = 0; q < size; q++){
                part_t p_i = (p + q) % size;
                if(cdata.conn_entries(start + p_i) == p){
                    p_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
        } else {
            b_con = cdata.conn_vals(start + best);
            p_con = cdata.conn_vals(start + p);
        }
        gain_update += p_con - b_con;
    }, cc_part1);
    Kokkos::parallel_for("perform moves", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        cdata.dest_cache(i) = NULL_PART;
        part_t p = part(i);
        part_t best = dest_part(i);
        dest_part(i) = p;
        part(i) = best;
        Kokkos::atomic_add(&part_sizes(p), -vtx_w(i));
        Kokkos::atomic_add(&part_sizes(best), vtx_w(i));
    });
    Kokkos::parallel_for("mark adjacent", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        //calculate gains for next phase
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            if(swap_bit(v) == 0) swap_bit(v) = 1;
        });
    });
    Kokkos::parallel_for("reset conn DS", team_policy_t(g.numRows(), Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(k*sizeof(gain_t) + k*sizeof(part_t) + 4*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = t.league_rank();
        if(swap_bit(i) == 1){
            edge_offset_t g_start = cdata.conn_offsets(i);
            edge_offset_t g_end = cdata.conn_offsets(i + 1);
            part_t size = g_end - g_start;
            cdata.dest_cache(i) = NULL_PART;
            gain_t* s_conn_vals = (gain_t*) t.team_shmem().get_shmem(sizeof(gain_t) * size);
            part_t* s_conn_entries = (part_t*) t.team_shmem().get_shmem(sizeof(part_t) * size);
            part_t* used_cap = (part_t*) t.team_shmem().get_shmem(sizeof(part_t));
            *used_cap = 0;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                s_conn_vals[j] = 0;
                cdata.conn_vals(g_start + j) = 0;
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                s_conn_entries[j] = NULL_PART;
                cdata.conn_entries(g_start + j) = NULL_PART;
            });
            t.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j){
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                part_t p = part(v);
                part_t p_o = p % size;
                if(size == k){
                    if(s_conn_entries[p_o] == NULL_PART && Kokkos::atomic_compare_exchange_strong(s_conn_entries + p_o, NULL_PART, p)) Kokkos::atomic_add(used_cap, 1);
                } else {
                    bool success = false;
                    while(!success){
                        while(s_conn_entries[p_o] != p && s_conn_entries[p_o] != NULL_PART){
                            p_o = (p_o + 1) % size;
                        }
                        if(s_conn_entries[p_o] == p){
                            success = true;
                        } else {
                            if(Kokkos::atomic_compare_exchange_strong(s_conn_entries + p_o, NULL_PART, p)) Kokkos::atomic_add(used_cap, 1);
                            if(s_conn_entries[p_o] == p){
                                success = true;
                            } else {
                                p_o = (p_o + 1) % size;
                            }
                        }
                    }
                }
                Kokkos::atomic_add(s_conn_vals + p_o, wgt);
            });
            t.team_barrier();
            part_t old_size = size;
            size = *used_cap;
            part_t quarter_size = size / 4;
            part_t min_inc = 3;
            if(quarter_size < min_inc) quarter_size = min_inc;
            size += quarter_size;
            if(size < old_size){
                cdata.conn_table_sizes(i) = size;
                //copy hashmap into reduced-size hashmap in global memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, old_size), [&] (const edge_offset_t& j){
                    part_t p = s_conn_entries[j];
                    if(p > NULL_PART){
                        part_t p_o = p % size;
                        bool success = false;
                        while(!success){
                            while(cdata.conn_entries(g_start + p_o) != NULL_PART){
                                p_o = (p_o + 1) % size;
                            }
                            Kokkos::atomic_compare_exchange(&cdata.conn_entries(g_start + p_o), NULL_PART, p);
                            if(cdata.conn_entries(g_start + p_o) == p){
                                success = true;
                            } else {
                                p_o = (p_o + 1) % size;
                            }
                        }
                        cdata.conn_vals(g_start + p_o) = s_conn_vals[j];
                    }
                });
            } else {
                size = old_size;
                cdata.conn_table_sizes(i) = size;
                //copy hashmap into full-size hashmap in global memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                    cdata.conn_vals(g_start + j) = s_conn_vals[j];
                });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                    cdata.conn_entries(g_start + j) = s_conn_entries[j];
                });
            }
            Kokkos::single(Kokkos::PerTeam(t), [=](){
                swap_bit(i) = 0;
            });
        }
    });
    gain_t cut_change_p2 = 0;
    Kokkos::parallel_reduce("count cutsize change part2", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        //swapped these earlier to save space
        part_t p = dest_part(i);
        part_t best = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        part_t p_con = 0;
        part_t b_con = 0;
        if(size < k){
            for(part_t q = 0; q < size; q++){
                part_t p_i = (best + q) % size;
                if(cdata.conn_entries(start + p_i) == best){
                    b_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
            for(part_t q = 0; q < size; q++){
                part_t p_i = (p + q) % size;
                if(cdata.conn_entries(start + p_i) == p){
                    p_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
        } else {
            b_con = cdata.conn_vals(start + best);
            p_con = cdata.conn_vals(start + p);
        }
        gain_update += b_con - p_con;
        if(x == 0) gain_update -= cc_part1();
    }, cut_change);
}

//7 kernels, 2 device-host syncs
void swap_and_update(const matrix_t g, const part_t k, const wgt_view_t vtx_w, part_vt part, const vtx_view_t swaps, const part_vt dest_part, scratch_mem& scratch, conn_data cdata, gain_t& cut_change, gain_vt part_sizes){
    ordinal_t total_moves = swaps.extent(0);
    cut_change = 0;
    vtx_view_t swap_bit = scratch.zeros1;
    //this reduction effectively takes place over two separate kernels
    //in order to avoid an extra copy to host, we store the first result on the device
    //and add it into the second result during the second kernel later
    //so that both can be transferred in a single copy to host
    gain_svt cc_part1 = scratch.cc_part1;
    Kokkos::parallel_reduce("count cutsize change part1", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        part_t p_con = 0;
        part_t b_con = 0;
        if(size < k){
            for(part_t q = 0; q < size; q++){
                part_t p_i = (best + q) % size;
                if(cdata.conn_entries(start + p_i) == best){
                    b_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
            for(part_t q = 0; q < size; q++){
                part_t p_i = (p + q) % size;
                if(cdata.conn_entries(start + p_i) == p){
                    p_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
        } else {
            b_con = cdata.conn_vals(start + best);
            p_con = cdata.conn_vals(start + p);
        }
        cdata.dest_cache(i) = NULL_PART;
        gain_update += p_con - b_con;
    }, cc_part1);
    Kokkos::parallel_for("update conns (subtract) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        part_t p = part(i);
        //calculate gains for next phase
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = p % v_size;
            if(v_size < k){
                //v is always adjacent to p because it is adjacent to i which is in p
                while(cdata.conn_entries(v_start + p_o) != p){
                    p_o = (p_o + 1) % v_size;
                }
            }
            //DO NOT USE ADD_FETCH HERE IT IS WAY SLOWER
            Kokkos::atomic_add(&cdata.conn_vals(v_start + p_o), -wgt);
            if(v_size < k && cdata.conn_vals(v_start + p_o) == 0){
                //free this gain slot
                cdata.conn_entries(v_start + p_o) = HASH_RECLAIM;
            }
        });
    });
    ordinal_t t_redos = 0;
    vtx_svt t_redos_dev = scratch.counter1;
    Kokkos::deep_copy(exec_space(), t_redos_dev, 0);
    vtx_view_t redos = scratch.vtx3;
    Kokkos::parallel_for("update conns (add) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        part_t best = dest_part(i);
        //calculate gains for next phase
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            cdata.dest_cache(v) = NULL_PART;
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = best % v_size;
            bool success = false;
            if(v_size == k){
                success = true;
                if(cdata.conn_entries(v_start + p_o) != best){
                    cdata.conn_entries(v_start + p_o) = best;
                }
            } else {
                for(part_t q = 0; q < v_size; q++){
                    part_t p_i = (best + q) % v_size;
                    if(cdata.conn_entries(v_start + p_i) == best){
                        success = true;
                        p_o = p_i;
                        break;
                    } else if(cdata.conn_entries(v_start + p_i) == NULL_PART){
                        break;
                    }
                }
            }
            part_t count = 0;
            while(!success && count < v_size){
                while(cdata.conn_entries(v_start + p_o) != best && cdata.conn_entries(v_start + p_o) > NULL_PART && count++ < v_size){
                    p_o = (p_o + 1) % v_size;
                }
                if(cdata.conn_entries(v_start + p_o) == best){
                    success = true;
                } else {
                    part_t orig = NULL_PART;
                    if(cdata.conn_entries(v_start + p_o) == HASH_RECLAIM) orig = HASH_RECLAIM;
                    //don't care if this thread succeeds if another thread succeeds with the same value
                    Kokkos::atomic_compare_exchange(&cdata.conn_entries(v_start + p_o), orig, best);
                    if(cdata.conn_entries(v_start + p_o) == best){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % v_size;
                        count++;
                    }
                }
            }
            if(success){
                Kokkos::atomic_add(&cdata.conn_vals(v_start + p_o), wgt);
            } else {
                //only a small number of rows should need to be fixed
                //a scan would be overkill for this
                ordinal_t add_redo = Kokkos::atomic_fetch_add(&swap_bit(v), 1);
                if(add_redo == 0){
                    ordinal_t insert = Kokkos::atomic_fetch_add(&t_redos_dev(), 1);
                    redos(insert) = v;
                }
            }
        });
    });
    Kokkos::deep_copy(exec_space(), t_redos, t_redos_dev);
    //unset swap bit, find imbalance change
    Kokkos::parallel_for("perform moves", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        part_t p = part(i);
        part_t best = dest_part(i);
        Kokkos::atomic_add(&part_sizes(p), -vtx_w(i));
        Kokkos::atomic_add(&part_sizes(best), vtx_w(i));
        part(i) = best;
        dest_part(i) = p;
    });
    //printf("Total rows to redo: %u\n", t_redos);
    Kokkos::parallel_for("fix overcapacity errors", team_policy_t(t_redos, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = redos(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        part_t size = g_end - g_start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_start, g_end), [&] (const edge_offset_t& j){
            cdata.conn_vals(j) = 0;
            cdata.conn_entries(j) = NULL_PART;
        });
        t.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            part_t p = part(v);
            if(size == k){
                Kokkos::atomic_add(&cdata.conn_vals(g_start + p), wgt);
                cdata.conn_entries(g_start + p) = p;
            } else {
                part_t p_o = p % size;
                bool success = false;
                while(!success){
                    while(cdata.conn_entries(g_start + p_o) != p && cdata.conn_entries(g_start + p_o) != NULL_PART){
                        p_o = (p_o + 1) % size;
                    }
                    if(cdata.conn_entries(g_start + p_o) == p){
                        success = true;
                    } else {
                        //don't care if this thread succeeds if another thread succeeds with the same value
                        Kokkos::atomic_compare_exchange(&cdata.conn_entries(g_start + p_o), NULL_PART, p);
                        if(cdata.conn_entries(g_start + p_o) == p){
                            success = true;
                        } else {
                            p_o = (p_o + 1) % size;
                        }
                    }
                }
                Kokkos::atomic_add(&cdata.conn_vals(g_start + p_o), wgt);
            }
        });
        t.team_barrier();
        Kokkos::single(Kokkos::PerTeam(t), [=](){
            cdata.dest_cache(i) = NULL_PART;
            swap_bit(i) = 0;
            cdata.conn_table_sizes(i) = size;
        });
    });
    gain_t cut_change_p2 = 0;
    Kokkos::parallel_reduce("count cutsize change part2", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        part_t p = dest_part(i);
        part_t best = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        part_t p_con = 0;
        part_t b_con = 0;
        if(size < k){
            for(part_t q = 0; q < size; q++){
                part_t p_i = (best + q) % size;
                if(cdata.conn_entries(start + p_i) == best){
                    b_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
            for(part_t q = 0; q < size; q++){
                part_t p_i = (p + q) % size;
                if(cdata.conn_entries(start + p_i) == p){
                    p_con = cdata.conn_vals(start + p_i);
                    break;
                } else if(cdata.conn_entries(start + p_i) == NULL_PART){
                    break;
                }
            }
        } else {
            b_con = cdata.conn_vals(start + best);
            p_con = cdata.conn_vals(start + p);
        }
        gain_update += b_con - p_con;
        if(x == 0) gain_update -= cc_part1();
    }, cut_change);
} 

gain_t largest_part_size(const gain_vt& ps){
    gain_t result = 0;
    Kokkos::parallel_reduce("get max part size", policy_t(0, ps.extent(0)), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
        if(ps(i) > update){
            update = ps(i);
        }
    }, Kokkos::Max<gain_t, Kokkos::HostSpace>(result));
    return result;
}

conn_data init_conn_data(const conn_data& scratch_cdata, const matrix_t& g, const part_vt& part, part_t k){
    ordinal_t n = g.numRows();
    conn_data cdata;
    cdata.conn_offsets = Kokkos::subview(scratch_cdata.conn_offsets, std::make_pair(static_cast<ordinal_t>(0), n + 1));
    Kokkos::parallel_for("comp conn row size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree > static_cast<ordinal_t>(k)) degree = k;
        cdata.conn_offsets(i + 1) = degree;
    });
    edge_offset_t gain_size = 0;
    Kokkos::parallel_scan("comp conn offsets", policy_t(0, n + 1), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update, const bool final){
        update += cdata.conn_offsets(i);
        if(final){
            cdata.conn_offsets(i) = update;
        }
    }, gain_size);
    cdata.conn_vals = Kokkos::subview(scratch_cdata.conn_vals, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.conn_entries = Kokkos::subview(scratch_cdata.conn_entries, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.dest_cache = Kokkos::subview(scratch_cdata.dest_cache, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.conn_table_sizes = Kokkos::subview(scratch_cdata.conn_table_sizes, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.lock_bit = Kokkos::subview(scratch_cdata.lock_bit, std::make_pair(static_cast<ordinal_t>(0), n));
    Kokkos::deep_copy(exec_space(), cdata.conn_vals, 0);
    Kokkos::deep_copy(exec_space(), cdata.conn_entries, NULL_PART);
    Kokkos::deep_copy(exec_space(), cdata.dest_cache, NULL_PART);
    Kokkos::deep_copy(exec_space(), cdata.lock_bit, 0);
    //find initial gain value for each vertex
    if(is_host_space || (g.nnz() / g.numRows()) < 8) {
        //printf("CPU version\n");
        //perform this on CPU
        Kokkos::parallel_for("init conn DS", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t& i){
            edge_offset_t g_start = cdata.conn_offsets(i);
            edge_offset_t g_end = cdata.conn_offsets(i + 1);
            part_t size = g_end - g_start;
            part_t used_cap = 0;
            for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                part_t p = part(v);
                part_t p_o = p % size;
                if(size < k){
                    while(cdata.conn_entries(g_start + p_o) != NULL_PART && cdata.conn_entries(g_start + p_o) != p){
                        p_o = (p_o + 1) % size;
                    }
                }
                cdata.conn_vals(g_start + p_o) += wgt;
                if(cdata.conn_entries(g_start + p_o) == NULL_PART){
                    cdata.conn_entries(g_start + p_o) = p;
                    used_cap++;
                }
            }
            part_t old_size = size;
            size = used_cap;
            part_t quarter_size = size / 4;
            part_t min_inc = 3;
            if(quarter_size < min_inc) quarter_size = min_inc;
            size += quarter_size;
            if(size < old_size){
                //don't get fancy just redo it with a smaller hashmap
                for(edge_offset_t j = g_start; j < g_start + size; j++){
                    cdata.conn_entries(j) = NULL_PART;
                    cdata.conn_vals(j) = 0;
                }
                for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                    ordinal_t v = g.graph.entries(j);
                    gain_t wgt = g.values(j);
                    part_t p = part(v);
                    part_t p_o = p % size;
                    if(size < k){
                        while(cdata.conn_entries(g_start + p_o) != NULL_PART && cdata.conn_entries(g_start + p_o) != p){
                            p_o = (p_o + 1) % size;
                        }
                    }
                    cdata.conn_vals(g_start + p_o) += wgt;
                    if(cdata.conn_entries(g_start + p_o) == NULL_PART){
                        cdata.conn_entries(g_start + p_o) = p;
                    }
                }
            } else {
                size = old_size;
            }
            cdata.conn_table_sizes(i) = size;
        });
    } else {
        //perform this on GPU
        //add 4*sizeof(part_t) for alignment reasons I think
        Kokkos::parallel_for("init conn DS (team)", team_policy_t(g.numRows(), Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(k*sizeof(gain_t) + k*sizeof(part_t) + 4*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
            ordinal_t i = t.league_rank();
            edge_offset_t g_start = cdata.conn_offsets(i);
            edge_offset_t g_end = cdata.conn_offsets(i + 1);
            part_t size = g_end - g_start;
            gain_t* s_conn_vals = (gain_t*) t.team_shmem().get_shmem(sizeof(gain_t) * size);
            part_t* s_conn_entries = (part_t*) t.team_shmem().get_shmem(sizeof(part_t) * size);
            part_t* used_cap = (part_t*) t.team_shmem().get_shmem(sizeof(part_t));
            *used_cap = 0;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                s_conn_vals[j] = 0;
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                s_conn_entries[j] = NULL_PART;
            });
            t.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j){
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                part_t p = part(v);
                part_t p_o = p % size;
                if(size == k){
                    if(s_conn_entries[p_o] == NULL_PART && Kokkos::atomic_compare_exchange_strong(s_conn_entries + p_o, NULL_PART, p)) Kokkos::atomic_add(used_cap, 1);
                } else {
                    bool success = false;
                    while(!success){
                        while(s_conn_entries[p_o] != p && s_conn_entries[p_o] != NULL_PART){
                            p_o = (p_o + 1) % size;
                        }
                        if(s_conn_entries[p_o] == p){
                            success = true;
                        } else {
                            if(Kokkos::atomic_compare_exchange_strong(s_conn_entries + p_o, NULL_PART, p)) Kokkos::atomic_add(used_cap, 1);
                            if(s_conn_entries[p_o] == p){
                                success = true;
                            } else {
                                p_o = (p_o + 1) % size;
                            }
                        }
                    }
                }
                Kokkos::atomic_add(s_conn_vals + p_o, wgt);
            });
            t.team_barrier();
            part_t old_size = size;
            size = *used_cap;
            part_t quarter_size = size / 4;
            part_t min_inc = 3;
            if(quarter_size < min_inc) quarter_size = min_inc;
            size += quarter_size;
            if(size < old_size){
                cdata.conn_table_sizes(i) = size;
                //copy hashmap into reduced-size hashmap in global memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, old_size), [&] (const edge_offset_t& j){
                    part_t p = s_conn_entries[j];
                    if(p > NULL_PART){
                        part_t p_o = p % size;
                        bool success = false;
                        while(!success){
                            while(cdata.conn_entries(g_start + p_o) != NULL_PART){
                                p_o = (p_o + 1) % size;
                            }
                            Kokkos::atomic_compare_exchange(&cdata.conn_entries(g_start + p_o), NULL_PART, p);
                            if(cdata.conn_entries(g_start + p_o) == p){
                                success = true;
                            } else {
                                p_o = (p_o + 1) % size;
                            }
                        }
                        cdata.conn_vals(g_start + p_o) = s_conn_vals[j];
                    }
                });
            } else {
                size = old_size;
                cdata.conn_table_sizes(i) = size;
                //copy hashmap into full-size hashmap in global memory
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                    cdata.conn_vals(g_start + j) = s_conn_vals[j];
                });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                    cdata.conn_entries(g_start + j) = s_conn_entries[j];
                });
            }
        });
    }
    return cdata;
}

void jet_refine(const matrix_t g, const part_t k, const double imb_ratio, wgt_view_t vtx_w, part_vt part, int level, refine_data& rfd, ExperimentLoggerUtil<scalar_t>& experiment){
    Kokkos::Timer y;
    //initialize metadata if this is first level being refined
    //ie. if this is the coarsest level
    //contains several scratch views that are reused in each iteration
    //reallocating in each iteration would be expensive
    scratch_mem& scratch = perm_scratch;
    if(!rfd.init){
        rfd.init = true;
        rfd.cut = get_total_cut(g, part);
        rfd.part_sizes = get_part_sizes(g, vtx_w, part, k);
        rfd.total_size = get_total_size(g, vtx_w);
        gain_t opt_size = rfd.total_size / k;
        gain_t max_size = largest_part_size(rfd.part_sizes);
        rfd.total_imb = max_size > opt_size ? max_size - opt_size : 0;
        std::cout << "Initial " << std::fixed << (rfd.cut / 2) << " " << (static_cast<double>(rfd.total_imb) / static_cast<double>(opt_size)) << " ";
        std::cout << g.numRows() << std::endl;
    }
    gain_vt part_sizes(Kokkos::ViewAllocateWithoutInitializing("part sizes"), k);
    Kokkos::deep_copy(exec_space(), part_sizes, rfd.part_sizes);
    gain_t cut = rfd.cut;
    gain_t opt_size = rfd.total_size / k;
    //imb_max is the maximum allowed difference
    //between largest part size and optimal part size
    gain_t imb_max = (imb_ratio - 1.0)*opt_size;
    ordinal_t n = g.numRows();
    //data about current best partition
    gain_t best_cut = cut;
    gain_t best_part_imb = rfd.total_imb;
    part_vt best_part(Kokkos::ViewAllocateWithoutInitializing("best partitioning encountered"), g.numRows());
    Kokkos::deep_copy(exec_space(), best_part, part);
    gain_t total_imb = best_part_imb;
    //initialize state machine
    conn_data cdata = init_conn_data(perm_cdata, g, part, k);
    int count = 0;
    int iter_count = 0;
    Kokkos::Timer iter_t;
    int balance_counter = 0;
    int lab_counter = 0;
    //repeat until 12 phases since a significant
    //improvement in cut or balance
    //this accounts for at least 3 full lp+rebalancing cycles
    while(count++ <= 11){
        iter_count++;
        gain_t cut_change = 0;
        vtx_view_t swaps;
        if(total_imb <= imb_max){
            //Kokkos::Timer t;
            swaps = jet_lp(g, k, part, cdata, scratch, (level == 0));
            balance_counter = 0;
            lab_counter++;
        } else {
            int t_buckets = max_buckets;
            //get vertices in a semi-sorted order
            if(balance_counter < 2){
                swaps = rebalance_weak(g, k, vtx_w, part, cdata, t_buckets, opt_size, imb_ratio, scratch, part_sizes);
            } else {
                swaps = rebalance_strong(g, k, vtx_w, part, cdata, t_buckets, opt_size, imb_ratio, scratch, part_sizes);
            }
            balance_counter++;
        }
        //perform swaps, update gains, and compute change to cut and imbalance
        if(static_cast<ordinal_t>(swaps.extent(0)) > static_cast<ordinal_t>(g.numRows() / 10)){
            swap_and_update_alt(g, k, vtx_w, part, swaps, scratch.dest_part, scratch.zeros1, cdata, cut_change, part_sizes, scratch);
        } else {
            swap_and_update(g, k, vtx_w, part, swaps, scratch.dest_part, scratch, cdata, cut_change, part_sizes);
        }
        gain_t max_size = largest_part_size(part_sizes);
        total_imb = max_size > opt_size ? max_size - opt_size : 0;
        cut = cut - cut_change;
        //copy current partition and relevant data to output partition if following conditions pass
        if(abs(best_part_imb) > imb_max && abs(total_imb) < abs(best_part_imb)){
            best_part_imb = total_imb;
            best_cut = cut;
            Kokkos::deep_copy(exec_space(), best_part, part);
            Kokkos::deep_copy(exec_space(), rfd.part_sizes, part_sizes);
            count = 0;
        } else if(cut < best_cut && (abs(total_imb) <= imb_max || abs(total_imb) <= abs(best_part_imb))){
            //do not reset counter if cut improvement is too small
            if(cut < 0.999*best_cut){
                count = 0;
            }
            best_cut = cut;
            best_part_imb = total_imb;
            Kokkos::deep_copy(exec_space(), best_part, part);
            Kokkos::deep_copy(exec_space(), rfd.part_sizes, part_sizes);
        }
    }
    //copy best partition to output
    rfd.total_imb = best_part_imb;
    rfd.cut = best_cut;
    Kokkos::deep_copy(exec_space(), part, best_part);
    Kokkos::fence();
    double best_imb_ratio = static_cast<double>(best_part_imb) / static_cast<double>(opt_size);
    typename ExperimentLoggerUtil<scalar_t>::CoarseLevel cl(best_cut / 2, best_imb_ratio, g.nnz(), g.numRows(), y.seconds(), iter_t.seconds(), iter_count, lab_counter);
    experiment.addCoarseLevel(cl);
    y.reset();
    iter_t.reset();
}
};

}
