#define NOMINMAX
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace py = pybind11;

#ifdef SOUP_WITH_CUDA_NF4
void decode_nf4_bf16_pair(
    int device, std::uintptr_t stream, std::uintptr_t lookup,
    std::uintptr_t gate_codes, std::uintptr_t gate_scales, std::uintptr_t gate_result,
    std::size_t gate_bytes, std::uintptr_t down_codes, std::uintptr_t down_scales,
    std::uintptr_t down_result, std::size_t down_bytes);
#endif

namespace {
using namespace pybind11::literals;
using U64 = unsigned long long;

constexpr U64 kSector = 4096;
constexpr U64 kMaxU64 = std::numeric_limits<U64>::max();
constexpr size_t kMaxExperts = 65536;
constexpr size_t kDefaultRingSlots = 4;
constexpr size_t kMaxIoWorkers = 16;
constexpr size_t kMaxRingSlots = 64;

U64 checked_add(U64 a, U64 b) {
    if (a > kMaxU64 - b) {
        throw std::overflow_error("native async stream byte accounting overflow");
    }
    return a + b;
}

class AlignedBuffer {
public:
    AlignedBuffer() = default;

    explicit AlignedBuffer(size_t size, size_t alignment = kSector) : size_(size) {
        if (size == 0) {
            return;
        }
#ifdef _WIN32
        ptr_ = static_cast<unsigned char*>(_aligned_malloc(size, alignment));
        if (!ptr_) {
            throw std::bad_alloc();
        }
#else
        void* raw = nullptr;
        if (posix_memalign(&raw, alignment, size) != 0 || !raw) {
            throw std::bad_alloc();
        }
        ptr_ = static_cast<unsigned char*>(raw);
#endif
    }

    ~AlignedBuffer() { reset(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void reset() noexcept {
        if (!ptr_) {
            return;
        }
#ifdef _WIN32
        _aligned_free(ptr_);
#else
        free(ptr_);
#endif
        ptr_ = nullptr;
        size_ = 0;
    }

    unsigned char* data() noexcept { return ptr_; }
    const unsigned char* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }

private:
    unsigned char* ptr_ = nullptr;
    size_t size_ = 0;
};

using Identity = std::array<U64, 5>;

Identity file_identity(int descriptor) {
#ifdef _WIN32
    const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(descriptor));
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(handle, &info)) {
        throw std::runtime_error("native expert file identity is unavailable");
    }
    return {
        (U64(info.nFileSizeHigh) << 32) | info.nFileSizeLow,
        (U64(info.nFileIndexHigh) << 32) | info.nFileIndexLow,
        info.dwVolumeSerialNumber,
        info.ftLastWriteTime.dwHighDateTime,
        info.ftLastWriteTime.dwLowDateTime,
    };
#else
    struct stat info {};
    if (fstat(descriptor, &info) != 0) {
        throw std::runtime_error("native expert file identity is unavailable");
    }
    return {
        U64(info.st_size),
        U64(info.st_ino),
        U64(info.st_dev),
#if defined(__APPLE__)
        // macOS spells the modification time st_mtimespec; Linux and BSD use st_mtim.
        U64(info.st_mtimespec.tv_sec),
        U64(info.st_mtimespec.tv_nsec),
#else
        U64(info.st_mtim.tv_sec),
        U64(info.st_mtim.tv_nsec),
#endif
    };
#endif
}

class ShardFile {
public:
    explicit ShardFile(int source, size_t worker_handles = 1) {
#ifdef _WIN32
        fd_ = _dup(source);
#else
        fd_ = fcntl(source, F_DUPFD_CLOEXEC, 0);
#endif
        if (fd_ < 0) {
            throw std::runtime_error("could not duplicate expert shard descriptor");
        }
        try {
#if defined(O_DIRECT)
            const int flags = fcntl(fd_, F_GETFL);
            if (flags < 0) {
                throw std::runtime_error("could not query expert shard read flags");
            }
            direct_io_ = (flags & O_DIRECT) != 0;
#endif
            initial_ = file_identity(fd_);
            if (initial_ != file_identity(source)) {
                throw std::runtime_error("expert shard changed during native registration");
            }
#ifdef _WIN32
            const auto base_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
            if (base_handle == INVALID_HANDLE_VALUE) {
                throw std::runtime_error("native expert Windows handle is unavailable");
            }
            worker_handles_.resize(std::max<size_t>(1, worker_handles), INVALID_HANDLE_VALUE);
            for (size_t idx = 0; idx < worker_handles_.size(); ++idx) {
                HANDLE reopened = ReOpenFile(
                    base_handle,
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    FILE_FLAG_RANDOM_ACCESS
                );
                if (reopened != INVALID_HANDLE_VALUE) {
                    worker_handles_[idx] = reopened;
                }
            }
#else
            (void)worker_handles;
#endif
        } catch (...) {
            close();
            throw;
        }
    }

    ~ShardFile() { close(); }
    ShardFile(const ShardFile&) = delete;
    ShardFile& operator=(const ShardFile&) = delete;

    void close() noexcept {
#ifdef _WIN32
        for (HANDLE h : worker_handles_) {
            if (h != INVALID_HANDLE_VALUE) {
                CloseHandle(h);
            }
        }
        worker_handles_.clear();
#endif
        if (fd_ < 0) {
            return;
        }
#ifdef _WIN32
        _close(fd_);
#else
        ::close(fd_);
#endif
        fd_ = -1;
    }

    U64 size() const noexcept { return initial_[0]; }

    void check() const {
        if (file_identity(fd_) != initial_) {
            throw std::runtime_error("expert shard changed after header validation");
        }
    }

    void read_exact(U64 offset, unsigned char* target, U64 capacity, U64 expected) const {
        read_exact_worker(0, offset, target, capacity, expected);
    }

    void read_exact_worker(
        size_t worker_id, U64 offset, unsigned char* target, U64 capacity, U64 expected
    ) const {
        U64 done = 0;
#ifdef _WIN32
        HANDLE handle = INVALID_HANDLE_VALUE;
        if (worker_id < worker_handles_.size() &&
            worker_handles_[worker_id] != INVALID_HANDLE_VALUE) {
            handle = worker_handles_[worker_id];
        } else {
            handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
        }
        while (done < expected) {
            const U64 pos = offset + done;
            const auto slice = std::min<U64>(capacity - done, 1ULL << 30);
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>(pos & 0xffffffffULL);
            ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
            DWORD got = 0;
            if (!ReadFile(handle, target + done, static_cast<DWORD>(slice), &got, &ov) ||
                got == 0) {
                throw std::runtime_error("failed or short read from expert shard");
            }
            done += static_cast<U64>(got);
        }
#else
        (void)worker_id;
        while (done < expected) {
            const auto got = pread(fd_, target + done, capacity - done, offset + done);
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got <= 0) {
                throw std::runtime_error("failed or short read from expert shard");
            }
            done += static_cast<U64>(got);
            // A direct read can end with a partial sector at EOF. If bytes are still
            // required, the next offset, length and destination must remain aligned.
            if (direct_io_ && done < expected && done % kSector != 0) {
                throw std::runtime_error("unaligned short O_DIRECT read from expert shard");
            }
        }
#endif
    }

private:
    int fd_ = -1;
    Identity initial_{};
    bool direct_io_ = false;
#ifdef _WIN32
    std::vector<HANDLE> worker_handles_;
#endif
};

struct Range {
    U64 start;
    U64 bytes;
    U64 lo;
    U64 hi;
    U64 packed_offset;
};

struct ExpertEntry {
    std::vector<Range> ranges;
    U64 payload_bytes = 0;
    std::unique_ptr<unsigned char[]> cached;
    bool admitting = false;
};

struct LayerEntry {
    LayerEntry(py::object reader_obj, int fd, size_t io_workers)
        : owner(std::move(reader_obj)), file(fd, io_workers) {}
    py::object owner;
    ShardFile file;
    std::vector<ExpertEntry> experts;
};

struct PrefetchItem {
    size_t seq_pos = 0;
    size_t expert_id = 0;
    bool cache_hit = false;
    size_t ring_slot = 0;
    const unsigned char* cached_ptr = nullptr;
    U64 fetches_delta = 0;
    U64 payload_delta = 0;
};

struct SequenceState {
    SequenceState(size_t layer, std::vector<size_t> experts, bool prefetch_flag)
        : layer_idx(layer),
          expert_ids(std::move(experts)),
          is_cross_layer_prefetch(prefetch_flag),
          ready_items(expert_ids.size()),
          ready_flags(expert_ids.size(), 0) {}

    size_t layer_idx = 0;
    std::vector<size_t> expert_ids;
    bool is_cross_layer_prefetch = false;
    size_t next_produce = 0;
    size_t next_consume = 0;
    size_t active_workers = 0;
    bool cancelled = false;
    std::vector<PrefetchItem> ready_items;
    std::vector<std::uint8_t> ready_flags;
    std::exception_ptr error = nullptr;
};

class PackedStreamAsync {
public:
    PackedStreamAsync(
        py::list registrations,
        U64 staging_bytes,
        U64 packed_bytes,
        U64 budget,
        size_t io_workers = 1,
        size_t ring_slots = kDefaultRingSlots
    )
        : staging_bytes_(staging_bytes),
          packed_bytes_(packed_bytes),
          budget_(budget),
          io_workers_(io_workers),
          ring_slots_(ring_slots) {
        if (staging_bytes_ == 0 || staging_bytes_ % kSector != 0 || packed_bytes_ == 0) {
            throw py::value_error("staging_bytes must be 4096-aligned and packed_bytes > 0");
        }
        if (io_workers_ == 0 || io_workers_ > kMaxIoWorkers) {
            throw py::value_error("io_workers must be between 1 and 16");
        }
        if (ring_slots_ < 2 || ring_slots_ > kMaxRingSlots) {
            throw py::value_error("ring_slots must be between 2 and 64");
        }
        scratches_.reserve(io_workers_);
        for (size_t idx = 0; idx < io_workers_; ++idx) {
            scratches_.emplace_back(static_cast<size_t>(staging_bytes_), kSector);
        }
        ring_.reserve(ring_slots_);
        ring_in_use_.assign(ring_slots_, false);
        for (size_t idx = 0; idx < ring_slots_; ++idx) {
            ring_.emplace_back(static_cast<size_t>(packed_bytes_), 256);
        }
        for (const auto item : registrations) {
            add_layer(item.cast<py::tuple>());
        }
        if (layers_.empty()) {
            throw py::value_error("no expert shards were registered");
        }
        workers_.reserve(io_workers_);
        for (size_t wid = 0; wid < io_workers_; ++wid) {
            workers_.emplace_back(&PackedStreamAsync::worker_loop, this, wid);
        }
    }

    ~PackedStreamAsync() { shutdown_noexcept(); }

    void fill_sync(size_t layer_idx, size_t expert_id, std::uintptr_t dst_addr) {
        unsigned char* dst = reinterpret_cast<unsigned char*>(dst_addr);
        if (!dst) {
            throw py::value_error("null destination pointer in fill_sync");
        }
        U64 fetches_delta = 0;
        U64 payload_delta = 0;
        const unsigned char* cached_src = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ensure_idle_locked();
            if (layer_idx >= layers_.size() || expert_id >= layers_[layer_idx]->experts.size()) {
                throw py::index_error("layer_idx or expert_id out of range");
            }
            busy_sync_ = true;
            ExpertEntry& exp = layers_[layer_idx]->experts[expert_id];
            if (exp.cached) {
                ++hits_;
                cached_src = exp.cached.get();
            } else {
                ++misses_;
            }
        }
        LayerEntry& layer = *layers_[layer_idx];
        ExpertEntry& expert = layer.experts[expert_id];
        try {
            if (cached_src != nullptr) {
                const auto t0 = Clock::now();
                {
                    py::gil_scoped_release release;
                    std::memcpy(dst, cached_src, static_cast<size_t>(packed_bytes_));
                }
                sync_hit_copy_ns_ += elapsed_ns(t0);
            } else {
                {
                    py::gil_scoped_release release;
                    read_and_pack_into(0, layer, expert, dst);
                }
                fetches_delta = 1;
                payload_delta = expert.payload_bytes;
                maybe_admit_cache(expert, dst);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            busy_sync_ = false;
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            busy_sync_ = false;
        }
        if (fetches_delta) {
            apply_reader_deltas(layer, fetches_delta, payload_delta);
        }
    }

    void start_sequence(size_t layer_idx, const std::vector<size_t>& expert_ids) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) {
            throw std::runtime_error("native async stream is closed");
        }
        if (busy_sync_ || cur_seq_ != nullptr) {
            throw std::runtime_error("native async stream already has an active operation");
        }
        validate_sequence_args_locked(layer_idx, expert_ids);

        if (next_seq_ != nullptr && next_seq_->layer_idx == layer_idx &&
            next_seq_->expert_ids == expert_ids) {
            cur_seq_ = std::move(next_seq_);
            cur_seq_->is_cross_layer_prefetch = false;
            ++cross_layer_promotions_;
            cv_workers_.notify_all();
            return;
        }

        if (next_seq_ != nullptr) {
            cancel_sequence_locked(lock, next_seq_);
            ++cross_layer_discards_;
        }

        std::fill(ring_in_use_.begin(), ring_in_use_.end(), false);
        cur_seq_ = std::make_shared<SequenceState>(layer_idx, expert_ids, false);
        cv_workers_.notify_all();
    }

    void enqueue_next_sequence(size_t layer_idx, const std::vector<size_t>& expert_ids) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) {
            throw std::runtime_error("native async stream is closed");
        }
        if (busy_sync_) {
            throw std::runtime_error("cannot enqueue cross-layer prefetch during sync fill");
        }
        validate_sequence_args_locked(layer_idx, expert_ids);
        if (next_seq_ != nullptr) {
            cancel_sequence_locked(lock, next_seq_);
        }
        if (!expert_ids.empty()) {
            next_seq_ = std::make_shared<SequenceState>(layer_idx, expert_ids, true);
            cv_workers_.notify_all();
        }
    }

    size_t consume_next(std::uintptr_t dst_addr) {
        unsigned char* dst = reinterpret_cast<unsigned char*>(dst_addr);
        if (!dst) {
            throw py::value_error("null destination pointer in consume_next");
        }
        PrefetchItem item{};
        size_t layer_idx = 0;
        {
            py::gil_scoped_release release;
            const auto wait_start = Clock::now();
            std::unique_lock<std::mutex> lock(mutex_);
            if (cur_seq_ == nullptr || cur_seq_->next_consume >= cur_seq_->expert_ids.size()) {
                throw std::runtime_error("no active prefetch item to consume");
            }
            std::shared_ptr<SequenceState> seq = cur_seq_;
            const size_t want_pos = seq->next_consume;
            layer_idx = seq->layer_idx;
            if (seq->ready_flags[want_pos]) {
                ++ring_immediate_ready_;
            }
            cv_main_.wait(lock, [&]() {
                return seq->ready_flags[want_pos] || seq->error != nullptr || closed_;
            });
            main_wait_worker_ns_ += elapsed_ns(wait_start);
            if (seq->error) {
                std::rethrow_exception(seq->error);
            }
            if (closed_) {
                throw std::runtime_error("native async stream closed while consuming");
            }
            item = seq->ready_items[want_pos];
            seq->ready_flags[want_pos] = 0;
            ++seq->next_consume;
            lock.unlock();

            const auto copy_start = Clock::now();
            const unsigned char* src =
                item.cache_hit ? item.cached_ptr : ring_[item.ring_slot].data();
            std::memcpy(dst, src, static_cast<size_t>(packed_bytes_));
            main_pinned_copy_ns_ += elapsed_ns(copy_start);

            if (!item.cache_hit) {
                lock.lock();
                ring_in_use_[item.ring_slot] = false;
                lock.unlock();
                cv_workers_.notify_all();
            }
        }
        if (item.fetches_delta) {
            apply_reader_deltas(*layers_[layer_idx], item.fetches_delta, item.payload_delta);
        }
        return item.expert_id;
    }

    void finish_sequence() {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(mutex_);
        if (cur_seq_ == nullptr) {
            return;
        }
        cancel_sequence_locked(lock, cur_seq_);
        if (next_seq_ != nullptr) {
            cv_workers_.notify_all();
        }
    }

    void cancel_next_sequence() {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(mutex_);
        if (next_seq_ != nullptr) {
            cancel_sequence_locked(lock, next_seq_);
        }
    }

    py::dict stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return py::dict(
            "budget_bytes"_a = budget_,
            "payload_bytes"_a = payload_,
            "entries"_a = entries_,
            "hits"_a = hits_,
            "misses"_a = misses_,
            "rejected"_a = rejected_
        );
    }

    py::dict native_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return py::dict(
            "io_workers"_a = io_workers_,
            "ring_slots"_a = ring_slots_,
            "range_reads"_a = read_calls_.load(),
            "requested_range_bytes"_a = physical_request_bytes_.load(),
            "worker_read_ns"_a = worker_read_ns_.load(),
            "worker_pack_ns"_a = worker_pack_ns_.load(),
            "worker_admit_ns"_a = worker_admit_ns_.load(),
            "sync_hit_copy_ns"_a = sync_hit_copy_ns_.load(),
            "main_wait_worker_ns"_a = main_wait_worker_ns_.load(),
            "main_pinned_copy_ns"_a = main_pinned_copy_ns_.load(),
            "ring_immediate_ready"_a = ring_immediate_ready_.load(),
            "cross_layer_prefetched"_a = cross_layer_prefetched_.load(),
            "cross_layer_promotions"_a = cross_layer_promotions_.load(),
            "cross_layer_discards"_a = cross_layer_discards_.load()
        );
    }

    void reset_native_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_idle_locked();
        read_calls_ = 0;
        physical_request_bytes_ = 0;
        worker_read_ns_ = 0;
        worker_pack_ns_ = 0;
        worker_admit_ns_ = 0;
        sync_hit_copy_ns_ = 0;
        main_wait_worker_ns_ = 0;
        main_pinned_copy_ns_ = 0;
        ring_immediate_ready_ = 0;
        cross_layer_prefetched_ = 0;
        cross_layer_promotions_ = 0;
        cross_layer_discards_ = 0;
    }

    void close() {
        finish_sequence();
        cancel_next_sequence();
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            if (busy_sync_) {
                throw std::runtime_error("cannot close native stream during active sync fill");
            }
            closed_ = true;
            cv_workers_.notify_all();
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        layers_.clear();
    }

private:
    using Clock = std::chrono::steady_clock;

    static U64 elapsed_ns(Clock::time_point start) {
        return static_cast<U64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
        );
    }

    void ensure_idle_locked() const {
        if (closed_) {
            throw std::runtime_error("native async stream is closed");
        }
        if (busy_sync_ || cur_seq_ != nullptr || next_seq_ != nullptr) {
            throw std::runtime_error("native async stream already has an active operation");
        }
    }

    void validate_sequence_args_locked(size_t layer_idx, const std::vector<size_t>& expert_ids) {
        if (layer_idx >= layers_.size()) {
            throw py::index_error("layer_idx out of range");
        }
        const size_t limit = layers_[layer_idx]->experts.size();
        for (size_t eid : expert_ids) {
            if (eid >= limit) {
                throw py::index_error("expert_id out of range in sequence");
            }
        }
    }

    void cancel_sequence_locked(
        std::unique_lock<std::mutex>& lock, std::shared_ptr<SequenceState>& seq_ref
    ) {
        if (!seq_ref) {
            return;
        }
        std::shared_ptr<SequenceState> seq = seq_ref;
        seq->cancelled = true;
        cv_workers_.notify_all();
        cv_main_.wait(lock, [&]() { return seq->active_workers == 0; });
        for (size_t pos = seq->next_consume; pos < seq->ready_flags.size(); ++pos) {
            if (seq->ready_flags[pos] && !seq->ready_items[pos].cache_hit) {
                ring_in_use_[seq->ready_items[pos].ring_slot] = false;
            }
        }
        seq_ref.reset();
    }

    void apply_reader_deltas(LayerEntry& layer, U64 fetches_delta, U64 payload_delta) {
        layer.owner.attr("fetches") = layer.owner.attr("fetches").cast<U64>() + fetches_delta;
        layer.owner.attr("bytes_read") =
            layer.owner.attr("bytes_read").cast<U64>() + payload_delta;
    }

    void read_and_pack_into(
        size_t worker_id, LayerEntry& layer, const ExpertEntry& expert, unsigned char* dst
    ) {
        layer.file.check();
        unsigned char* scratch = scratches_[worker_id].data();
        for (const Range& range : expert.ranges) {
            const auto r0 = Clock::now();
            const U64 actual = std::min(range.hi, layer.file.size()) - range.lo;
            layer.file.read_exact_worker(worker_id, range.lo, scratch, range.hi - range.lo, actual);
            worker_read_ns_ += elapsed_ns(r0);

            const auto p0 = Clock::now();
            std::memcpy(
                dst + range.packed_offset,
                scratch + (range.start - range.lo),
                static_cast<size_t>(range.bytes)
            );
            worker_pack_ns_ += elapsed_ns(p0);

            ++read_calls_;
            physical_request_bytes_ += actual;
        }
    }

    void maybe_admit_cache(ExpertEntry& expert, const unsigned char* packed_src) {
        if (!budget_) {
            return;
        }
        bool should_admit = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto t0 = Clock::now();
            if (!expert.cached && !expert.admitting && packed_bytes_ <= budget_ - payload_) {
                expert.admitting = true;
                payload_ += packed_bytes_;
                ++entries_;
                should_admit = true;
            } else if (!expert.cached && !expert.admitting) {
                ++rejected_;
            }
            worker_admit_ns_ += elapsed_ns(t0);
        }
        if (!should_admit) {
            return;
        }
        const auto t1 = Clock::now();
        auto block = std::make_unique<unsigned char[]>(static_cast<size_t>(packed_bytes_));
        std::memcpy(block.get(), packed_src, static_cast<size_t>(packed_bytes_));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            expert.cached = std::move(block);
            expert.admitting = false;
        }
        worker_admit_ns_ += elapsed_ns(t1);
    }

    size_t count_free_ring_slots_locked() const {
        size_t free_slots = 0;
        for (bool used : ring_in_use_) {
            if (!used) {
                ++free_slots;
            }
        }
        return free_slots;
    }

    int find_free_ring_slot_locked() const {
        for (size_t idx = 0; idx < ring_slots_; ++idx) {
            if (!ring_in_use_[idx]) {
                return static_cast<int>(idx);
            }
        }
        return -1;
    }

    bool can_take_sequence_locked(const std::shared_ptr<SequenceState>& seq, bool is_next) const {
        if (!seq || seq->cancelled || seq->error != nullptr) {
            return false;
        }
        if (seq->next_produce >= seq->expert_ids.size()) {
            return false;
        }
        if (is_next && cur_seq_ != nullptr &&
            cur_seq_->next_produce < cur_seq_->expert_ids.size()) {
            return false;
        }
        const auto& exp = layers_[seq->layer_idx]->experts[seq->expert_ids[seq->next_produce]];
        if (exp.cached != nullptr) {
            return true;
        }
        if (is_next && cur_seq_ != nullptr) {
            return count_free_ring_slots_locked() > 1;
        }
        return find_free_ring_slot_locked() >= 0;
    }

    void worker_loop(size_t worker_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!closed_) {
            cv_workers_.wait(lock, [&]() {
                return closed_ || can_take_sequence_locked(cur_seq_, false) ||
                    can_take_sequence_locked(next_seq_, true);
            });
            if (closed_) {
                break;
            }

            std::shared_ptr<SequenceState> seq;
            if (can_take_sequence_locked(cur_seq_, false)) {
                seq = cur_seq_;
            } else if (can_take_sequence_locked(next_seq_, true)) {
                seq = next_seq_;
            } else {
                continue;
            }

            const size_t pos = seq->next_produce;
            const size_t layer_idx = seq->layer_idx;
            const size_t expert_id = seq->expert_ids[pos];
            LayerEntry& layer = *layers_[layer_idx];
            ExpertEntry& expert = layer.experts[expert_id];

            if (expert.cached != nullptr) {
                ++hits_;
                PrefetchItem item{};
                item.seq_pos = pos;
                item.expert_id = expert_id;
                item.cache_hit = true;
                item.cached_ptr = expert.cached.get();
                ++seq->next_produce;
                seq->ready_items[pos] = item;
                seq->ready_flags[pos] = 1;
                if (seq->is_cross_layer_prefetch) {
                    ++cross_layer_prefetched_;
                }
                cv_main_.notify_all();
                continue;
            }

            const int slot = find_free_ring_slot_locked();
            if (slot < 0) {
                continue;
            }
            ring_in_use_[static_cast<size_t>(slot)] = true;
            ++seq->active_workers;
            ++seq->next_produce;
            ++misses_;
            unsigned char* target = ring_[static_cast<size_t>(slot)].data();
            lock.unlock();

            std::exception_ptr err = nullptr;
            try {
                read_and_pack_into(worker_id, layer, expert, target);
            } catch (...) {
                err = std::current_exception();
            }

            if (!err) {
                maybe_admit_cache(expert, target);
            }

            lock.lock();
            --seq->active_workers;
            if (err) {
                ring_in_use_[static_cast<size_t>(slot)] = false;
                if (!seq->error) {
                    seq->error = err;
                }
                cv_main_.notify_all();
                continue;
            }
            if (seq->cancelled) {
                ring_in_use_[static_cast<size_t>(slot)] = false;
                cv_main_.notify_all();
                continue;
            }
            PrefetchItem item{};
            item.seq_pos = pos;
            item.expert_id = expert_id;
            item.cache_hit = false;
            item.ring_slot = static_cast<size_t>(slot);
            item.fetches_delta = 1;
            item.payload_delta = expert.payload_bytes;
            seq->ready_items[pos] = item;
            seq->ready_flags[pos] = 1;
            if (seq->is_cross_layer_prefetch) {
                ++cross_layer_prefetched_;
            }
            cv_main_.notify_all();
        }
    }

    void shutdown_noexcept() noexcept {
        try {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                closed_ = true;
                if (cur_seq_) {
                    cur_seq_->cancelled = true;
                }
                if (next_seq_) {
                    next_seq_->cancelled = true;
                }
            }
            cv_workers_.notify_all();
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        } catch (...) {
        }
    }

    void add_layer(const py::tuple& row) {
        auto layer = std::make_unique<LayerEntry>(
            row[0].cast<py::object>(), row[1].cast<int>(), io_workers_);
        const py::list experts = row[2].cast<py::list>();
        if (!experts.size() || expert_count_ + experts.size() > kMaxExperts) {
            throw py::value_error("expert registry exceeds bounded size");
        }
        for (const auto item : experts) {
            ExpertEntry expert;
            for (const auto field : item.cast<py::list>()) {
                const auto range = field.cast<py::tuple>();
                const U64 start = range[0].cast<U64>();
                const U64 end = range[1].cast<U64>();
                const U64 offset = range[2].cast<U64>();
                if (start >= end || end > layer->file.size() ||
                    checked_add(offset, end - start) > packed_bytes_) {
                    throw py::value_error("expert metadata exceeds shard or packed buffer");
                }
                const U64 lo = start - start % kSector;
                const U64 hi = checked_add(end, kSector - 1) / kSector * kSector;
                if (hi - lo > staging_bytes_) {
                    throw py::value_error("expert range exceeds aligned staging capacity");
                }
                expert.ranges.push_back({start, end - start, lo, hi, offset});
                expert.payload_bytes += end - start;
            }
            if (expert.ranges.empty()) {
                throw py::value_error("expert cannot have empty ranges");
            }
            layer->experts.push_back(std::move(expert));
        }
        expert_count_ += experts.size();
        layers_.push_back(std::move(layer));
    }

    U64 staging_bytes_ = 0;
    U64 packed_bytes_ = 0;
    U64 budget_ = 0;
    size_t io_workers_ = 1;
    size_t ring_slots_ = kDefaultRingSlots;
    U64 expert_count_ = 0;
    std::vector<AlignedBuffer> scratches_;
    std::vector<AlignedBuffer> ring_;
    std::vector<bool> ring_in_use_;
    std::vector<std::unique_ptr<LayerEntry>> layers_;

    U64 payload_ = 0;
    U64 entries_ = 0;
    U64 hits_ = 0;
    U64 misses_ = 0;
    U64 rejected_ = 0;

    std::atomic<U64> read_calls_{0};
    std::atomic<U64> physical_request_bytes_{0};
    std::atomic<U64> worker_read_ns_{0};
    std::atomic<U64> worker_pack_ns_{0};
    std::atomic<U64> worker_admit_ns_{0};
    std::atomic<U64> sync_hit_copy_ns_{0};
    std::atomic<U64> main_wait_worker_ns_{0};
    std::atomic<U64> main_pinned_copy_ns_{0};
    std::atomic<U64> ring_immediate_ready_{0};
    std::atomic<U64> cross_layer_prefetched_{0};
    std::atomic<U64> cross_layer_promotions_{0};
    std::atomic<U64> cross_layer_discards_{0};

    mutable std::mutex mutex_;
    std::condition_variable cv_workers_;
    std::condition_variable cv_main_;
    std::vector<std::thread> workers_;

    bool closed_ = false;
    bool busy_sync_ = false;
    std::shared_ptr<SequenceState> cur_seq_;
    std::shared_ptr<SequenceState> next_seq_;
};

}  // namespace

PYBIND11_MODULE(_native_expert_stream_async, module) {
#ifdef SOUP_WITH_CUDA_NF4
    module.def("decode_nf4_bf16_pair", &decode_nf4_bf16_pair);
#endif
    py::class_<PackedStreamAsync>(module, "PackedStreamAsync")
        .def(
            py::init<py::list, U64, U64, U64, size_t, size_t>(),
            py::arg("registrations"),
            py::arg("staging_bytes"),
            py::arg("packed_bytes"),
            py::arg("budget"),
            py::arg("io_workers") = 1,
            py::arg("ring_slots") = kDefaultRingSlots
        )
        .def("fill_sync", &PackedStreamAsync::fill_sync)
        .def("start_sequence", &PackedStreamAsync::start_sequence)
        .def("enqueue_next_sequence", &PackedStreamAsync::enqueue_next_sequence)
        .def("cancel_next_sequence", &PackedStreamAsync::cancel_next_sequence)
        .def("consume_next", &PackedStreamAsync::consume_next)
        .def("finish_sequence", &PackedStreamAsync::finish_sequence)
        .def("stats", &PackedStreamAsync::stats)
        .def("native_stats", &PackedStreamAsync::native_stats)
        .def("reset_native_stats", &PackedStreamAsync::reset_native_stats)
        .def("close", &PackedStreamAsync::close);
}
