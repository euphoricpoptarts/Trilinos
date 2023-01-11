#pragma once
#include <list>
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_HashmapAccumulator.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#include "ExperimentLoggerUtil.hpp"
#include "heuristics.hpp"

namespace jet_partitioner {

template<typename ordinal_t>
KOKKOS_INLINE_FUNCTION ordinal_t xorshiftHash(ordinal_t key) {
  ordinal_t x = key;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

template<class crsMat> //typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class contracter {
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
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using pool_t = Kokkos::Random_XorShift64_Pool<Device>;
    using coarse_map = typename coarsen_heuristics<matrix_t>::coarse_map;
    static constexpr ordinal_t get_null_val() {
        // this value must line up with the null value used by the hashmap
        // accumulator
        if (std::is_signed<ordinal_t>::value) {
            return -1;
        } else {
            return std::numeric_limits<ordinal_t>::max();
        }
    }
    static constexpr ordinal_t ORD_MAX  = get_null_val();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    // contains matrix and vertex weights corresponding to current level
    // interp matrix maps previous level to this level
    struct coarse_level_triple {
        matrix_t mtx;
        wgt_view_t vtx_w;
        coarse_map interp_mtx;
        int level;
        bool uniform_weights;
    };

    // define behavior-controlling enums
    enum Heuristic { HECv1, HECv2, HECv3, Match, MtMetis, MIS2, GOSHv1, GOSHv2 };
    enum Builder { Sort, Hashmap_serial, Hashmap_parallel, Spgemm, Spgemm_transpose_first };

    // internal parameters and data
    // default heuristic is HEC
    Heuristic h = HECv1;
    // default builder is parallel hashmap
    Builder b = Hashmap_parallel;
    coarsen_heuristics<matrix_t> mapper;
    ordinal_t coarse_vtx_cutoff = 1000;
    ordinal_t min_allowed_vtx = 250;
    unsigned int max_levels = 200;

bool should_use_dyn(const ordinal_t n, const Kokkos::View<const edge_offset_t*, Device> work, int t_count){
    bool use_dyn = false;
    edge_offset_t max = 0;
    edge_offset_t min = std::numeric_limits<edge_offset_t>::max();
    if(is_host_space){
        ordinal_t static_size = (n + t_count) / t_count;
        for(ordinal_t i = 0; i < t_count; i++){
            ordinal_t start = i * static_size;
            ordinal_t end = start + static_size;
            if(start > n) start = n;
            if(end > n) end = n;
            edge_offset_t size = work(end) - work(start);
            if(size > max){
                max = size;
            }
            if(size < min) {
                min = size;
            }
        }
        //printf("min size: %i, max size: %i\n", min, max);
        if(n > 500000 && max > 5*min){
            use_dyn = true;
        }
    }
    return use_dyn;
}

template<typename uniform_memory_pool_t>
struct functorHashmapAccumulator
{

    typedef ordinal_t value_type;
    vtx_view_t remaining;
    edge_view_t row_map;
    vtx_view_t entries_in, entries_out;
    wgt_view_t wgts_in, wgts_out;
    edge_view_t dedupe_edge_count;
    uniform_memory_pool_t memory_pool;
    const ordinal_t hash_size;
    const ordinal_t max_hash_entries;

    functorHashmapAccumulator(edge_view_t _row_map,
        vtx_view_t _entries_in, vtx_view_t _entries_out,
        wgt_view_t _wgts_in, wgt_view_t _wgts_out,
        edge_view_t _dedupe_edge_count,
        uniform_memory_pool_t _memory_pool,
        const ordinal_t _hash_size,
        const ordinal_t _max_hash_entries,
        vtx_view_t _remaining)
        : row_map(_row_map)
        , entries_in(_entries_in)
        , entries_out(_entries_out)
        , wgts_in(_wgts_in)
        , wgts_out(_wgts_out)
        , dedupe_edge_count(_dedupe_edge_count)
        , memory_pool(_memory_pool)
        , hash_size(_hash_size)
        , max_hash_entries(_max_hash_entries)
        , remaining(_remaining){}
    
    //reduces to find total number of rows that were too large
    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& rem_idx, ordinal_t& thread_sum) const
    {
        ordinal_t idx = remaining(rem_idx);
        typedef ordinal_t hash_size_type;
        typedef ordinal_t hash_key_type;
        typedef scalar_t hash_value_type;

        //can't do this row at current hashmap size
        if(row_map(idx + 1) - row_map(idx) >= max_hash_entries){

            thread_sum++;
            return;
        }
        volatile ordinal_t* ptr_temp = nullptr;
        ordinal_t t_id = rem_idx;
#ifdef OPENMP
        if(std::is_same<exec_space, Kokkos::OpenMP>::value){
            t_id = Kokkos::OpenMP::impl_hardware_thread_id();
        } else if(std::is_same<exec_space, Kokkos::Serial>::value){
            t_id = 0;
        }
#else
        if(std::is_same<exec_space, Kokkos::Serial>::value){
            t_id = 0;
        }
#endif
        while (nullptr == ptr_temp)
        {
            ptr_temp = (volatile ordinal_t*)(memory_pool.allocate_chunk(t_id));
        }
        if(ptr_temp == nullptr){
            return;
        }
        ordinal_t* ptr_memory_pool_chunk = (ordinal_t*)(ptr_temp);
        
        // These are updated by Hashmap_Accumulator insert functions.
        ordinal_t * used_hash_size = (ordinal_t*)(ptr_temp);
        ptr_temp++;
        ordinal_t * used_hash_count = (ordinal_t*)(ptr_temp);
        ptr_temp++;
            *used_hash_size = 0;
            *used_hash_count = 0;

        // hash function is hash_size-1 (note: hash_size must be a power of 2)
        ordinal_t hash_func_pow2 = hash_size - 1;


        // Set pointer to hash indices
        ordinal_t* used_hash_indices = (ordinal_t*)(ptr_temp);
        ptr_temp += hash_size;

        // Set pointer to hash begins
        ordinal_t* hash_begins = (ordinal_t*)(ptr_temp);
        ptr_temp += hash_size;

        // Set pointer to hash nexts
        ordinal_t* hash_nexts = (ordinal_t*)(ptr_temp);

        // Set pointer to hash keys
        ordinal_t* keys = (ordinal_t*) entries_out.data() + row_map(idx);

        // Set pointer to hash values
        scalar_t* values = (scalar_t*) wgts_out.data() + row_map(idx);
        
        KokkosKernels::Experimental::HashmapAccumulator<hash_size_type, hash_key_type, hash_value_type, KokkosKernels::Experimental::HashOpType::bitwiseAnd> 
            hash_map(hash_size, hash_func_pow2, hash_begins, hash_nexts, keys, values);


        for(edge_offset_t i = row_map(idx); i < row_map(idx + 1); i++){
            ordinal_t key = entries_in(i);
            scalar_t value = wgts_in(i);
            //there appear to be small errors (but not catastrophic somehow) in the deduplication
            //this is likely due to insertion of duplicate keys simultaneously (not valid for this hashmap)
            int r = hash_map.sequential_insert_into_hash_mergeAdd_TrackHashes(
                key,
                value,
                used_hash_size,
                used_hash_count,
                used_hash_indices);

            // Check return code
            if (r)
            {
                // insert should return nonzero if the insert failed, but for sequential_insert_into_hash_TrackHashes
                // the 'full' case is currently ignored, so r will always be 0.
            }
        };

        // Reset the Begins values to -1 before releasing the memory pool chunk.
        // If you don't do this the next thread that grabs this memory chunk will not work properly.
        for(ordinal_t i = 0; i < *used_hash_count; i++) {
            ordinal_t dirty_hash = used_hash_indices[i];
            //entries(insert_at) = hash_map.keys[i];
            //wgts(insert_at) = hash_map.values[i];

            hash_map.hash_begins[dirty_hash] = ORD_MAX;
            //insert_at++;
        };


        //used_hash_size gives the number of entries, used_hash_count gives the number of dirty hash values (I think)
        dedupe_edge_count(idx) = *used_hash_size;//insert_at - row_map(idx);
        // Release the memory pool chunk back to the pool
        memory_pool.release_chunk(ptr_memory_pool_chunk);

    }   // operator()

};  // functorHashmapAccumulator


void getHashmapSizeAndCount(const ordinal_t n, const ordinal_t remaining_count, vtx_view_t remaining, vtx_view_t edges_per_source, ordinal_t& hash_size, ordinal_t& max_entries, ordinal_t& mem_chunk_size, ordinal_t& mem_chunk_count){
    ordinal_t avg_entries = 0;
    if (!is_host_space && static_cast<double>(remaining_count) / static_cast<double>(n) > 0.01) {
        Kokkos::parallel_reduce("calc average among remaining", policy_t(0, remaining_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & thread_sum){
            ordinal_t u = remaining(i);
            ordinal_t degree = edges_per_source(u);
            thread_sum += degree;
        }, avg_entries);
        //degrees are often skewed so we want to err on the side of bigger hashmaps
        avg_entries = avg_entries * 2  / remaining_count;
        avg_entries++;
        if (avg_entries < 50) avg_entries = 50;
    }
    else {
        Kokkos::parallel_reduce("calc max", policy_t(0, remaining_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & thread_max){
            ordinal_t u = remaining(i);
            ordinal_t degree = edges_per_source(u);
            if (degree > thread_max) {
                thread_max = degree;
            }
        }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(avg_entries));
        //need precisely one larger than max, don't remember why atm
        avg_entries++;
    }

    // Set the hash_size as the next power of 2 bigger than hash_size_hint.
    // - hash_size must be a power of two since we use & rather than % (which is slower) for
    // computing the hash value for HashmapAccumulator.
    max_entries = avg_entries;
    hash_size = 1;
    while (hash_size < max_entries) { hash_size *= 2; }

    // Determine memory chunk size for UniformMemoryPool
    mem_chunk_size = hash_size;      // for hash indices
    mem_chunk_size += hash_size;            // for hash begins
    mem_chunk_size += 3*max_entries;     // for hash nexts, keys, and values
    mem_chunk_size += 10; // for metadata
    // Set a cap on # of chunks to 32.  In application something else should be done
    // here differently if we're OpenMP vs. GPU but for this example we can just cap
    // our number of chunks at 32.
    mem_chunk_count = exec_space::concurrency();
    if(mem_chunk_count > remaining_count){
        mem_chunk_count = remaining_count + 1;
    }

    if (!is_host_space) {
        //decrease number of mem_chunks to reduce memory usage if necessary
        size_t mem_needed = static_cast<size_t>(mem_chunk_count) * static_cast<size_t>(mem_chunk_size) * sizeof(ordinal_t);
        //size_t max_mem_allowed = 1073741824;
        size_t max_mem_allowed = 536870912;
        //size_t max_mem_allowed = 268435456;
        if (mem_needed > max_mem_allowed) {
            size_t chunk_dif = mem_needed - max_mem_allowed;
            chunk_dif = chunk_dif / (static_cast<size_t>(mem_chunk_size) * sizeof(ordinal_t));
            chunk_dif++;
            mem_chunk_count -= chunk_dif;
        }
    }
}

void deduplicate_graph(const ordinal_t n, const bool use_team, const bool is_coarse,
    vtx_view_t edges_per_source, vtx_view_t dest_by_source, wgt_view_t wgt_by_source,
    const edge_view_t source_bucket_offset, ExperimentLoggerUtil<scalar_t>& experiment, edge_offset_t& gc_nedges) {

        Kokkos::Timer radix;
        ordinal_t remaining_count = n;
        vtx_view_t remaining("remaining vtx", n);
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            remaining(i) = i;
        });
        do {
            //determine size for hashmap
            ordinal_t hash_size, max_entries, mem_chunk_size, mem_chunk_count;
            getHashmapSizeAndCount(n, remaining_count, remaining, edges_per_source, hash_size, max_entries, mem_chunk_size, mem_chunk_count);
            // Create Uniform Initialized Memory Pool
            KokkosKernels::Impl::PoolType pool_type = KokkosKernels::Impl::ManyThread2OneChunk;

            if (is_host_space) {
                pool_type = KokkosKernels::Impl::OneThread2OneChunk;
            }

            bool use_dyn = should_use_dyn(n, source_bucket_offset, mem_chunk_count);

            typedef typename KokkosKernels::Impl::UniformMemoryPool<exec_space, ordinal_t> uniform_memory_pool_t;
            uniform_memory_pool_t memory_pool(mem_chunk_count, mem_chunk_size, ORD_MAX, pool_type);

            functorHashmapAccumulator<uniform_memory_pool_t>
                hashmapAccumulator(source_bucket_offset, dest_by_source, dest_by_source, wgt_by_source, wgt_by_source, edges_per_source, memory_pool, hash_size, max_entries, remaining);

            ordinal_t old_remaining_count = remaining_count;
            experiment.addMeasurement(Measurement::HashmapAllocate, radix.seconds());
            radix.reset();
            //if(!is_host_space && max_entries >= 128){
            //    //printf("Team hashmap\n");
            //    Kokkos::parallel_reduce("hashmap time", team_policy_t(old_remaining_count, 1, 64), hashmapAccumulator, remaining_count);
            //} else {
                //printf("Sequential hashmap\n");
                if(use_dyn){
                    Kokkos::parallel_reduce("hashmap time", dyn_policy_t(0, old_remaining_count, Kokkos::ChunkSize(128)), hashmapAccumulator, remaining_count);
                } else {
                    Kokkos::parallel_reduce("hashmap time", policy_t(0, old_remaining_count), hashmapAccumulator, remaining_count);
                }
            //}
            //printf("Hashmap size: %u; rows deduped: %u\n", max_entries, old_remaining_count - remaining_count);
            //printf("Hashmap dedupe time: %.4f seconds\n", radix.seconds());
            experiment.addMeasurement(Measurement::HashmapInsert, radix.seconds());
            radix.reset();

            if (remaining_count > 0) {
                vtx_view_t new_remaining("new remaining vtx", remaining_count);

                Kokkos::parallel_scan("move remaining vertices", policy_t(0, old_remaining_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t & update, const bool final){
                    ordinal_t u = remaining(i);
                    if (edges_per_source(u) >= max_entries) {
                        if (final) {
                            new_remaining(update) = u;
                        }
                        update++;
                    }
                });

                remaining = new_remaining;
            }
            experiment.addMeasurement(Measurement::HashmapAllocate, radix.seconds());
            radix.reset();
            //printf("remaining count: %u\n", remaining_count);
        } while (remaining_count > 0);
        Kokkos::parallel_reduce(policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t & sum){
            sum += edges_per_source(i);
        }, gc_nedges);
}

struct countingFunctor {

    matrix_t g;
    vtx_view_t vcmap;
    edge_view_t degree_initial;
    wgt_view_t c_vtx_w, f_vtx_w;
    ordinal_t workLength;

    countingFunctor(matrix_t _g,
            vtx_view_t _vcmap,
            edge_view_t _degree_initial,
            wgt_view_t _c_vtx_w,
            wgt_view_t _f_vtx_w) :
        g(_g),
        vcmap(_vcmap),
        degree_initial(_degree_initial),
        c_vtx_w(_c_vtx_w),
        f_vtx_w(_f_vtx_w),
        workLength(_g.numRows()) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& i) const 
    {
        ordinal_t u = vcmap(i);
        edge_offset_t start = g.graph.row_map(i);
        edge_offset_t end = g.graph.row_map(i + 1);
        ordinal_t nonLoopEdgesTotal = end - start;
        Kokkos::atomic_add(&degree_initial(u), nonLoopEdgesTotal);
        Kokkos::atomic_add(&c_vtx_w(u), f_vtx_w(i));
    }
};

struct combineAndDedupe {
    matrix_t g;
    vtx_view_t vcmap;
    vtx_view_t htable;
    wgt_view_t hvals;
    edge_view_t hrow_map;

    combineAndDedupe(matrix_t _g,
            vtx_view_t _vcmap,
            vtx_view_t _htable,
            wgt_view_t _hvals,
            edge_view_t _hrow_map) :
            g(_g),
            vcmap(_vcmap),
            htable(_htable),
            hvals(_hvals),
            hrow_map(_hrow_map) {}

    KOKKOS_INLINE_FUNCTION
        edge_offset_t insert(const edge_offset_t& hash_start, const edge_offset_t& size, const ordinal_t& u) const {
            edge_offset_t offset = abs(xorshiftHash<ordinal_t>(u)) % size;
            while(true){
                if(htable(hash_start + offset) == -1){
                    Kokkos::atomic_compare_exchange(&htable(hash_start + offset), -1, u);
                }
                if(htable(hash_start + offset) == u){
                    return offset;
                } else {
                    offset++;
                    if(offset >= size) offset -= size;
                }
            }
        }

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t x = thread.league_rank();
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i != u){
                edge_offset_t offset = insert(hash_start, size, u);
                Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
            }
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& x) const
    {
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        for(edge_offset_t j = start; j < end; j++){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i != u){
                edge_offset_t offset = insert(hash_start, size, u);
                Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
            }
        }
    }
};

struct countUnique {
    vtx_view_t htable;
    edge_view_t hrow_map, coarse_row_map_f;

    countUnique(vtx_view_t _htable,
            edge_view_t _hrow_map,
            edge_view_t _coarse_row_map_f) :
            htable(_htable),
            hrow_map(_hrow_map),
            coarse_row_map_f(_coarse_row_map_f) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t i = thread.league_rank();
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        ordinal_t uniques = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j, ordinal_t& update){
            if(htable(j) != -1){
                update++;
            }
        }, uniques);
        Kokkos::single(Kokkos::PerTeam(thread), [=](){
            coarse_row_map_f(i) = uniques;
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t i) const
    {
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        ordinal_t uniques = 0;
        for(edge_offset_t j = start; j < end; j++) {
            if(htable(j) != -1){
                uniques++;
            }
        }
        coarse_row_map_f(i) = uniques;
    }
};

struct consolidateUnique {
    vtx_view_t htable, entries_coarse;
    wgt_view_t hvals, wgts_coarse;
    edge_view_t hrow_map, coarse_row_map_f;

    consolidateUnique(vtx_view_t _htable,
            vtx_view_t _entries_coarse,
            wgt_view_t _hvals,
            wgt_view_t _wgts_coarse,
            edge_view_t _hrow_map,
            edge_view_t _coarse_row_map_f) :
            htable(_htable),
            entries_coarse(_entries_coarse),
            hvals(_hvals),
            wgts_coarse(_wgts_coarse),
            hrow_map(_hrow_map),
            coarse_row_map_f(_coarse_row_map_f) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t i = thread.league_rank();
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        const edge_offset_t write_to = coarse_row_map_f(i);
        ordinal_t* total = (ordinal_t*) thread.team_shmem().get_shmem(sizeof(ordinal_t));
        *total = 0;
        thread.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            if(htable(j) != -1){
                //we don't care about the insertion order
                //this is faster than a scan
                ordinal_t insert = Kokkos::atomic_fetch_add(total, 1);
                entries_coarse(write_to + insert) = htable(j);
                wgts_coarse(write_to + insert) = hvals(j);
            }
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t i) const
    {
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        edge_offset_t write_to = coarse_row_map_f(i);
        for (edge_offset_t j = start; j < end; j++){
            if(htable(j) != -1){
                entries_coarse(write_to) = htable(j);
                wgts_coarse(write_to) = hvals(j);
                write_to++;
            }
        };
    }
};

coarse_level_triple build_coarse_graph(const coarse_level_triple level,
    const coarse_map vcmap,
    ExperimentLoggerUtil<scalar_t>& experiment) {

    matrix_t g = level.mtx;
    ordinal_t n = g.numRows();
    ordinal_t nc = vcmap.coarse_vtx;

    Kokkos::Timer timer;
    edge_view_t hrow_map("hashtable row map", nc + 1);
    wgt_view_t f_vtx_w = level.vtx_w;
    wgt_view_t c_vtx_w = wgt_view_t("coarse vertex weights", nc);
    countingFunctor countF(g, vcmap.map, hrow_map, c_vtx_w, f_vtx_w);
    Kokkos::parallel_for("count edges per coarse vertex (also compute coarse vertex weights)", policy_t(0, n), countF);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Count, timer.seconds());
    timer.reset();
    edge_offset_t hash_size = 0;
    //exclusive prefix sum
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = hrow_map(i);
        if(final){
            hrow_map(i) = update;
        }
        update += val;
    }, hash_size);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Prefix, timer.seconds());
    timer.reset();
    vtx_view_t htable(Kokkos::ViewAllocateWithoutInitializing("hashtable keys"), hash_size);
    Kokkos::deep_copy(htable, -1);
    wgt_view_t hvals("hashtable values", hash_size);
    //insert each coarse vertex into a bucket determined by a hash
    //use linear probing to resolve conflicts
    //combine weights using atomic addition
    combineAndDedupe cnd(g, vcmap.map, htable, hvals, hrow_map);
    if(!is_host_space && hash_size / n >= 12) {
        Kokkos::parallel_for("deduplicate", team_policy_t(n, Kokkos::AUTO), cnd);
    } else {
        bool use_dyn = should_use_dyn(n, g.graph.row_map, exec_space::concurrency());
        if(use_dyn){
            Kokkos::parallel_for("deduplicate", dyn_policy_t(0, n), cnd);
        } else {
            Kokkos::parallel_for("deduplicate", policy_t(0, n), cnd);
        }
    }
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Dedupe, timer.seconds());
    timer.reset();
    edge_view_t coarse_row_map_f("edges_per_source", nc + 1);
    countUnique cu(htable, hrow_map, coarse_row_map_f);
    if(!is_host_space && hash_size / nc >= 12) {
        Kokkos::parallel_for("count unique", team_policy_t(nc, Kokkos::AUTO), cu);
    } else {
        Kokkos::parallel_for("count unique", policy_t(0, nc), cu);
    }
    Kokkos::fence();
    experiment.addMeasurement(Measurement::WriteGraph, timer.seconds());
    timer.reset();
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = coarse_row_map_f(i);
        if(final){
            coarse_row_map_f(i) = update;
        }
        update += val;
    }, hash_size);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Prefix, timer.seconds());
    timer.reset();
    vtx_view_t entries_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse entries"), hash_size);
    wgt_view_t wgts_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse weights"), hash_size);
    consolidateUnique consolidate(htable, entries_coarse, hvals, wgts_coarse, hrow_map, coarse_row_map_f);
    if(!is_host_space && hash_size / nc >= 12) {
        Kokkos::parallel_for("consolidate", team_policy_t(nc, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(4*sizeof(ordinal_t))), consolidate);
    } else {
        bool use_dyn = should_use_dyn(nc, hrow_map, exec_space::concurrency());
        if(use_dyn){
            Kokkos::parallel_for("consolidate", dyn_policy_t(0, nc), consolidate);
        } else {
            Kokkos::parallel_for("consolidate", policy_t(0, nc), consolidate);
        }
    }
    graph_type gc_graph(entries_coarse, coarse_row_map_f);
    matrix_t gc("gc", nc, wgts_coarse, gc_graph);
    coarse_level_triple next_level;
    next_level.mtx = gc;
    next_level.vtx_w = c_vtx_w;
    next_level.level = level.level + 1;
    next_level.interp_mtx = vcmap;
    next_level.uniform_weights = false;
    Kokkos::fence();
    experiment.addMeasurement(Measurement::WriteGraph, timer.seconds());
    timer.reset();
    return next_level;
}

coarse_map generate_coarse_mapping(const matrix_t g,
    const wgt_view_t& vtx_w,
    bool uniform_weights,
    pool_t& rand_pool,
    ExperimentLoggerUtil<scalar_t>& experiment) {

    Kokkos::Timer timer;
    coarse_map interpolation_graph;
    int choice = 0;

    switch (h) {
        case HECv1:
            choice = 0;
            break;
        case HECv2:
            choice = 1;
            break;
        case HECv3:
            choice = 2;
            break;
        case Match:
            choice = 0;
            break;
        case MtMetis:
            choice = 1;
            break;
        default:
            choice = 0;
    }

    switch (h) {
        case HECv1:
        case HECv2:
        case HECv3:
            interpolation_graph = mapper.coarsen_HEC(g, vtx_w, uniform_weights, rand_pool, experiment);
            break;
        case Match:
        case MtMetis:
            interpolation_graph = mapper.coarsen_match(g, uniform_weights, rand_pool, experiment, choice);
            break;
        case MIS2:
            interpolation_graph = mapper.coarsen_mis_2(g);
            break;
        case GOSHv2:
            interpolation_graph = mapper.coarsen_GOSH_v2(g);
            break;
        case GOSHv1:
            interpolation_graph = mapper.coarsen_GOSH(g);
            break;
    }
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Map, timer.seconds());
    return interpolation_graph;
}

std::list<coarse_level_triple> generate_coarse_graphs(const matrix_t fine_g, const wgt_view_t vweights, ExperimentLoggerUtil<scalar_t>& experiment, bool uniform_eweights = false) {

    Kokkos::Timer timer;
    ordinal_t fine_n = fine_g.numRows();
    std::list<coarse_level_triple> levels;
    coarse_level_triple finest;
    finest.mtx = fine_g;
    //1-indexed, not zero indexed
    finest.level = 1;
    finest.uniform_weights = uniform_eweights;
    finest.vtx_w = vweights;
    levels.push_back(finest);
    pool_t rand_pool(std::time(nullptr));
    while (levels.rbegin()->mtx.numRows() > coarse_vtx_cutoff) {

        coarse_level_triple current_level = *levels.rbegin();

        coarse_map interp_graph = generate_coarse_mapping(current_level.mtx, current_level.vtx_w, current_level.uniform_weights, rand_pool, experiment);

        if (interp_graph.coarse_vtx < min_allowed_vtx) {
            break;
        }

        timer.reset();
        coarse_level_triple next_level = build_coarse_graph(current_level, interp_graph, experiment);
        Kokkos::fence();
        experiment.addMeasurement(Measurement::Build, timer.seconds());
        timer.reset();

        levels.push_back(next_level);

        if(levels.size() > max_levels) break;
#ifdef DEBUG
        double coarsen_ratio = (double) levels.rbegin()->mtx.numRows() / (double) (++levels.rbegin())->mtx.numRows();
        printf("Coarsening ratio: %.8f\n", coarsen_ratio);
#endif
    }
    return levels;
}

void set_heuristic(Heuristic _h) {
    this->h = _h;
}

void set_deduplication_method(Builder _b) {
    this->b = _b;
}

void set_coarse_vtx_cutoff(ordinal_t _coarse_vtx_cutoff) {
    this->coarse_vtx_cutoff = _coarse_vtx_cutoff;
}

void set_min_allowed_vtx(ordinal_t _min_allowed_vtx) {
    this->min_allowed_vtx = _min_allowed_vtx;
}

void set_max_levels(unsigned int _max_levels) {
    this->max_levels = _max_levels;
}

};

}
