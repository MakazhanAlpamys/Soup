#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kBytesPerThread = 4;
constexpr int kBytesPerBlock = kThreads * kBytesPerThread;
constexpr int kPackedBytesPerScale = 32;

struct Projection {
    const std::uint8_t* packed;
    const float* scales;
    std::uint32_t* decoded;
    std::size_t bytes;
    bool aligned;
};

__device__ std::uint16_t bf16_bits(float value) {
#if __CUDA_ARCH__ >= 800
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
#else
    // Round to nearest even on T4, where bfloat16 conversion is emulated.
    const std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return 0x7fffU;
    }
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
#endif
}

__device__ std::uint32_t decode_byte(std::uint8_t code, const float* lookup, float scale) {
    const auto high = bf16_bits(__fmul_rn(lookup[code >> 4], scale));
    const auto low = bf16_bits(__fmul_rn(lookup[code & 15], scale));
    // bitsandbytes stores the first weight in the high nibble.
    return static_cast<std::uint32_t>(high) | (static_cast<std::uint32_t>(low) << 16);
}

__global__ void decode_pair(
    const float* lookup, Projection gate, Projection down, unsigned int gate_blocks) {
    // Assign whole blocks to one projection so each warp reads a single weight array.
    const bool is_gate = blockIdx.x < gate_blocks;
    const Projection part = is_gate ? gate : down;
    const auto block = is_gate ? blockIdx.x : blockIdx.x - gate_blocks;
    const auto index = (static_cast<std::size_t>(block) * blockDim.x + threadIdx.x)
        * kBytesPerThread;
    if (index >= part.bytes) {
        return;
    }

    // A group of four packed bytes stays within one block-64 scale.
    const float scale = part.scales[index / kPackedBytesPerScale];
    if (part.aligned && part.bytes - index >= kBytesPerThread) {
        const auto input = reinterpret_cast<const std::uint32_t*>(part.packed)[index / 4];
        const uint4 decoded{
            decode_byte(static_cast<std::uint8_t>(input), lookup, scale),
            decode_byte(static_cast<std::uint8_t>(input >> 8), lookup, scale),
            decode_byte(static_cast<std::uint8_t>(input >> 16), lookup, scale),
            decode_byte(static_cast<std::uint8_t>(input >> 24), lookup, scale),
        };
        reinterpret_cast<uint4*>(part.decoded)[index / 4] = decoded;
    } else {
#pragma unroll
        for (int offset = 0; offset < kBytesPerThread; ++offset) {
            if (index + offset < part.bytes) {
                part.decoded[index + offset] =
                    decode_byte(part.packed[index + offset], lookup, scale);
            }
        }
    }
}

}  // namespace

void decode_nf4_bf16_pair(
    int device, std::uintptr_t stream, std::uintptr_t lookup,
    std::uintptr_t gate_codes, std::uintptr_t gate_scales, std::uintptr_t gate_result,
    std::size_t gate_bytes, std::uintptr_t down_codes, std::uintptr_t down_scales,
    std::uintptr_t down_result, std::size_t down_bytes) {
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    const Projection gate{
        reinterpret_cast<const std::uint8_t*>(gate_codes),
        reinterpret_cast<const float*>(gate_scales),
        reinterpret_cast<std::uint32_t*>(gate_result),
        gate_bytes,
        gate_codes % alignof(std::uint32_t) == 0 && gate_result % alignof(uint4) == 0,
    };
    const Projection down{
        reinterpret_cast<const std::uint8_t*>(down_codes),
        reinterpret_cast<const float*>(down_scales),
        reinterpret_cast<std::uint32_t*>(down_result),
        down_bytes,
        down_codes % alignof(std::uint32_t) == 0 && down_result % alignof(uint4) == 0,
    };
    const auto gate_blocks = static_cast<unsigned int>(
        (gate_bytes + kBytesPerBlock - 1) / kBytesPerBlock);
    const auto down_blocks = static_cast<unsigned int>(
        (down_bytes + kBytesPerBlock - 1) / kBytesPerBlock);
    decode_pair<<<gate_blocks + down_blocks, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const float*>(lookup), gate, down, gate_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
