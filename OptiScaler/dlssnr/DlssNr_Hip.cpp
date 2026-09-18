#include "pch.h"

#include "DlssNr_Hip.h"

#include <Config.h>
#include <Logger.h>
#include <Util.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// See DlssNr_Hip.h for what this is. In short: the same model, run by libdlss5_hip.so on the Linux
// side of the process, reached through the dlss5_hip.dll trampoline, because NGX cannot run it here.

namespace DlssNr
{
namespace Hip
{
namespace
{

// ---------------------------------------------------------------------------------------------
// The C ABI published by libdlss5_hip.so / dlss5_hip.dll.
//
// dlss5_run is the network and nothing else: packed float RGBA in, packed float RGB out, both
// 1920x1080, no encode and no composition. That is exactly the shape of the call this backend
// replaces -- the pass hands the model a proxy it has already built and composes the answer itself.

using PFN_Init = int (*)(const char* weightsDir, int gpu);
using PFN_Run = int (*)(const float* rgba, float* rgb, unsigned int seed);
using PFN_LastError = const char* (*) ();
using PFN_Shutdown = void (*)();

constexpr size_t kPixels = size_t(ModelWidth) * size_t(ModelHeight);

// ---------------------------------------------------------------------------------------------
// Pixel conversion.
//
// The staging copy carries the frame's own format, whatever the game renders in, and the model wants
// packed float32. Converting on the CPU costs a few milliseconds against the model's ~230, and it
// keeps this backend out of the pass's resource creation entirely -- no scratch texture changes
// format, no shader is rebuilt, and an unsupported format fails with a name rather than a picture.

float HalfToFloat(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            const uint32_t bits = sign;
            float out;
            std::memcpy(&out, &bits, 4);
            return out;
        }

        // Subnormal: normalise it by hand.
        while ((mantissa & 0x400u) == 0)
        {
            mantissa <<= 1;
            exponent--;
        }

        exponent++;
        mantissa &= 0x3FFu;
    }
    else if (exponent == 31)
    {
        const uint32_t bits = sign | 0x7F800000u | (mantissa << 13);
        float out;
        std::memcpy(&out, &bits, 4);
        return out;
    }

    const uint32_t bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, 4);

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t rawExponent = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;

    if (rawExponent == 0xFFu) // inf / NaN
        return uint16_t(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));

    const int32_t exponent = int32_t(rawExponent) - 112; // 127 - 15

    if (exponent >= 0x1F)
        return uint16_t(sign | 0x7BFFu); // saturate rather than hand the frame an infinity

    if (exponent <= 0)
    {
        if (exponent < -10)
            return uint16_t(sign);

        // Subnormal: shift the implied one back in and round to nearest even.
        const uint32_t withImplied = mantissa | 0x800000u;
        const uint32_t shift = uint32_t(14 - exponent);
        const uint32_t truncated = withImplied >> shift;
        const uint32_t half = 1u << (shift - 1);
        const uint32_t remainder = withImplied & ((1u << shift) - 1u);
        const uint32_t rounded =
            truncated + ((remainder > half || (remainder == half && (truncated & 1u) != 0)) ? 1u : 0u);
        return uint16_t(sign | rounded);
    }

    // Normal: round to nearest even, letting a carry roll into the exponent, which is exactly what
    // the field layout wants -- 0x3FF + 1 becomes the next exponent with a zero mantissa.
    const uint32_t packed = (uint32_t(exponent) << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1FFFu;
    const uint32_t rounded = packed + ((remainder > 0x1000u || (remainder == 0x1000u && (packed & 1u) != 0)) ? 1u : 0u);

    // A carry out of the top saturates the same way the overflow case above does.
    return uint16_t(sign | (rounded >= 0x7C00u ? 0x7BFFu : rounded));
}

float Float11ToFloat(uint32_t v) // R11G11B10_FLOAT: 5 exponent bits, 6 mantissa
{
    const uint32_t exponent = (v >> 6) & 0x1Fu;
    const uint32_t mantissa = v & 0x3Fu;
    const uint32_t bits = exponent == 0 ? 0u : ((exponent + 112u) << 23) | (mantissa << 17);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint32_t FloatToFloat11(float f)
{
    if (!(f > 0.0f))
        return 0;

    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 112;

    if (exponent <= 0)
        return 0;

    if (exponent >= 0x1F)
        return 0x7BFu;

    return (uint32_t(exponent) << 6) | ((bits >> 17) & 0x3Fu);
}

float Float10ToFloat(uint32_t v) // R11G11B10_FLOAT blue: 5 exponent bits, 5 mantissa
{
    const uint32_t exponent = (v >> 5) & 0x1Fu;
    const uint32_t mantissa = v & 0x1Fu;
    const uint32_t bits = exponent == 0 ? 0u : ((exponent + 112u) << 23) | (mantissa << 18);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint32_t FloatToFloat10(float f)
{
    if (!(f > 0.0f))
        return 0;

    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 112;

    if (exponent <= 0)
        return 0;

    if (exponent >= 0x1F)
        return 0x3DFu;

    return (uint32_t(exponent) << 5) | ((bits >> 18) & 0x1Fu);
}

float Saturate(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

bool FormatSupported(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return true;
    default:
        return false;
    }
}

// One row of the staged proxy, into the packed RGBA the model reads.
void DecodeRow(const uint8_t* src, float* dst, DXGI_FORMAT format, unsigned int width)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    {
        const auto* p = reinterpret_cast<const uint16_t*>(src);

        for (unsigned int x = 0; x < width; x++)
        {
            dst[x * 4 + 0] = HalfToFloat(p[x * 4 + 0]);
            dst[x * 4 + 1] = HalfToFloat(p[x * 4 + 1]);
            dst[x * 4 + 2] = HalfToFloat(p[x * 4 + 2]);
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    {
        const auto* p = reinterpret_cast<const float*>(src);

        for (unsigned int x = 0; x < width; x++)
        {
            dst[x * 4 + 0] = p[x * 4 + 0];
            dst[x * 4 + 1] = p[x * 4 + 1];
            dst[x * 4 + 2] = p[x * 4 + 2];
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    {
        const bool bgr = format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;

        for (unsigned int x = 0; x < width; x++)
        {
            const uint8_t r = src[x * 4 + (bgr ? 2 : 0)];
            const uint8_t g = src[x * 4 + 1];
            const uint8_t b = src[x * 4 + (bgr ? 0 : 2)];
            dst[x * 4 + 0] = float(r) / 255.0f;
            dst[x * 4 + 1] = float(g) / 255.0f;
            dst[x * 4 + 2] = float(b) / 255.0f;
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    {
        const auto* p = reinterpret_cast<const uint32_t*>(src);

        for (unsigned int x = 0; x < width; x++)
        {
            const uint32_t v = p[x];
            dst[x * 4 + 0] = float(v & 0x3FFu) / 1023.0f;
            dst[x * 4 + 1] = float((v >> 10) & 0x3FFu) / 1023.0f;
            dst[x * 4 + 2] = float((v >> 20) & 0x3FFu) / 1023.0f;
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case DXGI_FORMAT_R11G11B10_FLOAT:
    {
        const auto* p = reinterpret_cast<const uint32_t*>(src);

        for (unsigned int x = 0; x < width; x++)
        {
            const uint32_t v = p[x];
            dst[x * 4 + 0] = Float11ToFloat(v & 0x7FFu);
            dst[x * 4 + 1] = Float11ToFloat((v >> 11) & 0x7FFu);
            dst[x * 4 + 2] = Float10ToFloat((v >> 22) & 0x3FFu);
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    default:
        break;
    }
}

// The model's answer (packed RGB) back into one row of the frame's format. Alpha is written opaque:
// the answer is a colour image and the resolve reads colour from it.
void EncodeRow(const float* src, uint8_t* dst, DXGI_FORMAT format, unsigned int width)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    {
        auto* p = reinterpret_cast<uint16_t*>(dst);

        for (unsigned int x = 0; x < width; x++)
        {
            p[x * 4 + 0] = FloatToHalf(src[x * 3 + 0]);
            p[x * 4 + 1] = FloatToHalf(src[x * 3 + 1]);
            p[x * 4 + 2] = FloatToHalf(src[x * 3 + 2]);
            p[x * 4 + 3] = FloatToHalf(1.0f);
        }

        break;
    }
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    {
        auto* p = reinterpret_cast<float*>(dst);

        for (unsigned int x = 0; x < width; x++)
        {
            p[x * 4 + 0] = src[x * 3 + 0];
            p[x * 4 + 1] = src[x * 3 + 1];
            p[x * 4 + 2] = src[x * 3 + 2];
            p[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    {
        const bool bgr = format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;

        for (unsigned int x = 0; x < width; x++)
        {
            const auto r = uint8_t(Saturate(src[x * 3 + 0]) * 255.0f + 0.5f);
            const auto g = uint8_t(Saturate(src[x * 3 + 1]) * 255.0f + 0.5f);
            const auto b = uint8_t(Saturate(src[x * 3 + 2]) * 255.0f + 0.5f);
            dst[x * 4 + (bgr ? 2 : 0)] = r;
            dst[x * 4 + 1] = g;
            dst[x * 4 + (bgr ? 0 : 2)] = b;
            dst[x * 4 + 3] = 255;
        }

        break;
    }
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    {
        auto* p = reinterpret_cast<uint32_t*>(dst);

        for (unsigned int x = 0; x < width; x++)
        {
            const auto r = uint32_t(Saturate(src[x * 3 + 0]) * 1023.0f + 0.5f);
            const auto g = uint32_t(Saturate(src[x * 3 + 1]) * 1023.0f + 0.5f);
            const auto b = uint32_t(Saturate(src[x * 3 + 2]) * 1023.0f + 0.5f);
            p[x] = r | (g << 10) | (b << 20) | (3u << 30);
        }

        break;
    }
    case DXGI_FORMAT_R11G11B10_FLOAT:
    {
        auto* p = reinterpret_cast<uint32_t*>(dst);

        for (unsigned int x = 0; x < width; x++)
            p[x] = FloatToFloat11(src[x * 3 + 0]) | (FloatToFloat11(src[x * 3 + 1]) << 11) |
                   (FloatToFloat10(src[x * 3 + 2]) << 22);

        break;
    }
    default:
        break;
    }
}

// ---------------------------------------------------------------------------------------------
// The exchange.

struct Exchange
{
    // What the model is reached through.
    HMODULE library = nullptr;
    PFN_Init init = nullptr;
    PFN_Run run = nullptr;
    PFN_LastError lastError = nullptr;
    PFN_Shutdown shutdown = nullptr;
    bool modelReady = false;

    // The staging pair, and the frame geometry they were built for.
    ID3D12Device* device = nullptr;
    ID3D12Resource* readback = nullptr;
    ID3D12Resource* upload = nullptr;
    uint8_t* readbackMapped = nullptr;
    uint8_t* uploadMapped = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT64 stagedBytes = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;

    // Knowing when a recorded copy has actually run. A Signal issued while recording frame N is
    // ordered after every list already submitted, which includes the one that carried frame N-1's
    // copy -- so that copy is complete once the fence passes this tick, with nothing to guess.
    ID3D12Fence* fence = nullptr;
    UINT64 tick = 0;

    // State machine, all of it under `lock`.
    //   Idle       nothing in flight; the next frame may stage its proxy
    //   Recorded   a copy into the readback buffer is on a list that has not been submitted yet
    //   Submitted  that list is submitted; the copy completes when the fence reaches `readableAt`
    //   Running    the worker owns the buffers
    //   Ready      an answer is in the upload buffer, waiting to be copied over the output
    enum class Stage
    {
        Idle,
        Recorded,
        Submitted,
        Running,
        Ready
    };

    std::mutex lock;
    std::condition_variable wake;
    Stage stage = Stage::Idle;
    UINT64 readableAt = 0;   // fence tick at which the staged proxy is readable
    UINT64 writableAt = 0;   // fence tick after which the upload buffer may be rewritten
    bool haveAnswer = false; // the output texture holds a finished answer from some earlier frame
    bool quit = false;
    unsigned int seed = 0;
    float lastMs = 0.0f;
    std::thread worker;

    // Scratch, owned by the worker and sized once.
    std::vector<float> input;
    std::vector<float> answer;
};

Exchange g;
std::mutex g_once;       // serialises Ensure/Release against each other
std::mutex g_statusLock; // the worker reports failures too, so this is its own lock
std::string g_status = "not started";
std::atomic<bool> g_failed { false }; // sticky: a failed backend does not retry into a crash

// Deliberately does not take g_once: Release() holds that while joining the worker, and the worker
// is one of the callers.
void Fail(const std::string& why)
{
    g_failed = true;
    {
        std::lock_guard<std::mutex> held(g_statusLock);
        g_status = why;
    }
    LOG_ERROR("DLSS-NR (HIP): {}", why);
}

void Say(const std::string& what)
{
    std::lock_guard<std::mutex> held(g_statusLock);
    g_status = what;
}

std::wstring FindTrampoline()
{
    // Beside OptiScaler first, then beside the game, then whatever the environment names. The
    // trampoline is tiny and always shipped with the HIP build, so "beside OptiScaler" is the
    // normal answer.
    if (auto beside = Util::FindFilePath(Util::DllPath().remove_filename(), "dlss5_hip.dll"))
        return beside->wstring();

    if (auto beside = Util::FindFilePath(Util::ExePath().remove_filename(), "dlss5_hip.dll"))
        return beside->wstring();

    wchar_t fromEnv[MAX_PATH] {};

    if (GetEnvironmentVariableW(L"DLSS5_HIP_DLL", fromEnv, MAX_PATH) != 0)
        return fromEnv;

    return {};
}

std::string WeightsDirectory()
{
    // A Linux path: it is handed straight through the trampoline to libdlss5_hip.so, which is the
    // Linux side of this process and reads it with Linux file calls.
    char fromEnv[1024] {};

    if (GetEnvironmentVariableA("DLSS5_WEIGHTS", fromEnv, sizeof(fromEnv)) != 0)
        return fromEnv;

    return {};
}

void WorkerMain()
{
    HANDLE ready = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    for (;;)
    {
        UINT64 waitFor = 0;

        {
            std::unique_lock<std::mutex> held(g.lock);
            g.wake.wait(held, [] { return g.quit || g.stage == Exchange::Stage::Submitted; });

            if (g.quit)
                break;

            waitFor = g.readableAt > g.writableAt ? g.readableAt : g.writableAt;
            g.stage = Exchange::Stage::Running;
        }

        // Wait for the copy that staged the proxy, and for the GPU to be done with whatever the
        // upload buffer last carried.
        if (g.fence != nullptr && ready != nullptr)
        {
            if (g.fence->GetCompletedValue() < waitFor)
            {
                if (SUCCEEDED(g.fence->SetEventOnCompletion(waitFor, ready)))
                    WaitForSingleObject(ready, 5000);
            }
        }

        // Staged frame -> packed float RGBA.
        const auto rowPitch = size_t(g.footprint.Footprint.RowPitch);

        for (unsigned int y = 0; y < ModelHeight; y++)
            DecodeRow(g.readbackMapped + y * rowPitch, g.input.data() + size_t(y) * ModelWidth * 4, g.format,
                      ModelWidth);

        const auto started = std::chrono::steady_clock::now();
        const int result = g.run(g.input.data(), g.answer.data(), g.seed++);
        const auto ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();

        if (result != 0)
        {
            const char* why = g.lastError != nullptr ? g.lastError() : "unknown";
            std::lock_guard<std::mutex> held(g.lock);
            g.stage = Exchange::Stage::Idle;
            Fail(std::string("the model failed: ") + (why != nullptr ? why : "unknown"));
            continue;
        }

        // Answer -> the frame's format, in the upload buffer the next frame copies from.
        for (unsigned int y = 0; y < ModelHeight; y++)
            EncodeRow(g.answer.data() + size_t(y) * ModelWidth * 3, g.uploadMapped + y * rowPitch, g.format,
                      ModelWidth);

        {
            std::lock_guard<std::mutex> held(g.lock);
            g.lastMs = ms;
            g.stage = Exchange::Stage::Ready;
        }
    }

    if (ready != nullptr)
        CloseHandle(ready);
}

void Barrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, D3D12_RESOURCE_STATES from,
             D3D12_RESOURCE_STATES to)
{
    if (from == to)
        return;

    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    cmdList->ResourceBarrier(1, &barrier);
}

void CopyTextureToBuffer(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* texture, ID3D12Resource* buffer)
{
    D3D12_TEXTURE_COPY_LOCATION src {};
    src.pResource = texture;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dst {};
    dst.pResource = buffer;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = g.footprint;

    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

void CopyBufferToTexture(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* buffer, ID3D12Resource* texture)
{
    D3D12_TEXTURE_COPY_LOCATION src {};
    src.pResource = buffer;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = g.footprint;

    D3D12_TEXTURE_COPY_LOCATION dst {};
    dst.pResource = texture;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

// Whether this pointer answers like a command queue. Its own function because MSVC will not allow
// __try in a function that has objects to unwind, and Evaluate is full of them.
bool QueueLooksReal(ID3D12CommandQueue* queue)
{
    ID3D12CommandQueue* real = nullptr;
    bool usable = false;

    __try
    {
        usable = SUCCEEDED(queue->QueryInterface(IID_PPV_ARGS(&real))) && real != nullptr;
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        usable = false;
    }

    if (real != nullptr)
        real->Release();

    return usable;
}

// Every D3D12 call this backend makes on the game's own command list, in one place, with no C++
// object in scope so that structured exception handling is legal here.
//
// The first frame of this pass took the game down before its menu, and a crash inside a game's render
// thread says nothing about which call did it. Now each step is numbered, the number reaches the log,
// and an access violation disables the backend instead of ending the session: a bug of ours should
// cost the player the feature, not their game.
//
// Returns the step it reached (1 signal, 2-4 deliver, 5-6 stage); negative means it faulted there.
int GuardedRecord(ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* queue, ID3D12Resource* modelInput,
                  ID3D12Resource* output, unsigned long long tick, int deliver, int stage)
{
    int step = 0;

    __try
    {
        step = 1;
        queue->Signal(g.fence, tick);

        if (deliver)
        {
            step = 2;
            Barrier(cmdList, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            step = 3;
            CopyBufferToTexture(cmdList, g.upload, output);
            step = 4;
            Barrier(cmdList, output, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        if (stage)
        {
            step = 5;
            Barrier(cmdList, modelInput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyTextureToBuffer(cmdList, modelInput, g.readback);
            Barrier(cmdList, modelInput, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            step = 6;
        }
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return -step;
    }

    return step;
}

bool CreateStaging(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc)
{
    UINT rows = 0;
    UINT64 rowBytes = 0;
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &g.footprint, &rows, &rowBytes, &total);

    D3D12_RESOURCE_DESC bufferDesc {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = total;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    D3D12_HEAP_PROPERTIES readbackHeap {};
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_HEAP_PROPERTIES uploadHeap {};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

    if (FAILED(device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g.readback))) ||
        FAILED(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g.upload))))
    {
        Fail("the staging buffers could not be allocated");
        return false;
    }

    // Mapped for the life of the exchange: the worker writes and reads them off the render thread,
    // and D3D12 is explicit that a persistent map is the intended way to do that.
    D3D12_RANGE nothing { 0, 0 };

    if (FAILED(g.readback->Map(0, nullptr, reinterpret_cast<void**>(&g.readbackMapped))) ||
        FAILED(g.upload->Map(0, &nothing, reinterpret_cast<void**>(&g.uploadMapped))))
    {
        Fail("the staging buffers could not be mapped");
        return false;
    }

    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence))))
    {
        Fail("the fence could not be created");
        return false;
    }

    g.stagedBytes = total;
    g.format = desc.Format;
    g.input.assign(kPixels * 4, 0.0f);
    g.answer.assign(kPixels * 3, 0.0f);
    return true;
}

} // namespace

bool Enabled()
{
    // Off unless asked for, and asked for by the environment rather than the ini: this backend only
    // means anything inside a Proton process that has libdlss5_hip.so preloaded, and that is set up
    // by the same launch script that would set this.
    char value[64] {};

    if (GetEnvironmentVariableA("DLSS5_NR_BACKEND", value, sizeof(value)) == 0)
        return false;

    auto same = [](const char* a, const char* b)
    {
        for (; *a != 0 && *b != 0; a++, b++)
            if ((*a | 0x20) != (*b | 0x20))
                return false;

        return *a == *b;
    };

    return same(value, "hip") || same(value, "amd") || same(value, "1") || same(value, "true");
}

bool Ensure(ID3D12Device* device)
{
    std::lock_guard<std::mutex> once(g_once);

    if (g_failed)
        return false;

    if (g.modelReady)
        return true;

    if (device == nullptr)
        return false;

    const std::wstring path = FindTrampoline();

    if (path.empty())
    {
        Fail("dlss5_hip.dll was not found beside OptiScaler or the game (set DLSS5_HIP_DLL to point "
             "at it)");
        return false;
    }

    g.library = LoadLibraryW(path.c_str());

    if (g.library == nullptr)
    {
        Fail("dlss5_hip.dll would not load");
        return false;
    }

    g.init = (PFN_Init) GetProcAddress(g.library, "dlss5_init");
    g.run = (PFN_Run) GetProcAddress(g.library, "dlss5_run");
    g.lastError = (PFN_LastError) GetProcAddress(g.library, "dlss5_last_error");
    g.shutdown = (PFN_Shutdown) GetProcAddress(g.library, "dlss5_shutdown");

    if (g.init == nullptr || g.run == nullptr)
    {
        Fail("dlss5_hip.dll is missing its exports");
        return false;
    }

    const std::string weights = WeightsDirectory();

    if (weights.empty())
    {
        Fail("no weights directory: set DLSS5_WEIGHTS to the Linux path of the extracted weights");
        return false;
    }

    if (g.init(weights.c_str(), 0) != 0)
    {
        const char* why = g.lastError != nullptr ? g.lastError() : nullptr;
        Fail(std::string("the model would not initialise: ") + (why != nullptr ? why : "unknown"));
        return false;
    }

    g.device = device;
    g.modelReady = true;
    Say("ready");
    LOG_INFO("DLSS-NR (HIP): model initialised from {}, running at {}x{}", weights, ModelWidth, ModelHeight);
    return true;
}

int Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* queue, ID3D12Resource* modelInput,
             ID3D12Resource* output, unsigned int workWidth, unsigned int workHeight, bool reset)
{
    if (g_failed || cmdList == nullptr || modelInput == nullptr || output == nullptr)
        return 0;

    if (workWidth != ModelWidth || workHeight != ModelHeight)
    {
        Fail("the model runs at 1920x1080 only, and this frame asked for another size");
        return 0;
    }

    if (queue == nullptr)
    {
        // Without the queue there is no way to know when a copy has run, and reading a buffer the
        // GPU may not have written yet would show the model a frame of noise.
        Say("waiting for the command queue");
        return 0;
    }

    // The fallback queue arrives as a void* OptiScaler keeps for its timing; ask it whether it really
    // is a command queue before calling one. Signal on something that is not would jump through a
    // vtable that isn't there, on the game's render thread, on the first frame.
    {
        static ID3D12CommandQueue* vetted = nullptr;

        if (vetted != queue)
        {
            if (!QueueLooksReal(queue))
            {
                Fail("the command queue handed to the pass is not a usable ID3D12CommandQueue");
                return 0;
            }

            vetted = queue;
        }
    }

    const D3D12_RESOURCE_DESC desc = modelInput->GetDesc();

    if (!FormatSupported(desc.Format))
    {
        Fail("the frame's format is not one this backend can stage");
        return 0;
    }

    if (g.readback == nullptr)
    {
        ID3D12Device* device = nullptr;

        if (FAILED(modelInput->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
            return 0;

        const bool made = CreateStaging(device, desc);
        device->Release();

        if (!made)
            return 0;

        g.worker = std::thread(WorkerMain);
    }

    // One tick per frame, ordered after everything already submitted. See Exchange::fence.
    const UINT64 tick = ++g.tick;
    queue->Signal(g.fence, tick);

    if (reset)
    {
        std::lock_guard<std::mutex> held(g.lock);
        g.haveAnswer = false;
    }

    bool stage = false;
    bool deliver = false;

    {
        std::lock_guard<std::mutex> held(g.lock);

        // A copy recorded last frame is on a list that has now been submitted, so this tick is when
        // it will have run.
        if (g.stage == Exchange::Stage::Recorded)
        {
            g.stage = Exchange::Stage::Submitted;
            g.readableAt = tick;
            g.wake.notify_one();
        }
        else if (g.stage == Exchange::Stage::Ready)
        {
            deliver = true;
            g.stage = Exchange::Stage::Idle;
            g.writableAt = tick + 1; // the copy below runs no later than the next tick
            g.haveAnswer = true;
        }
        else if (g.stage == Exchange::Stage::Idle)
        {
            stage = true;
            g.stage = Exchange::Stage::Recorded;
        }
    }

    const int reached = GuardedRecord(cmdList, queue, modelInput, output, tick, deliver ? 1 : 0, stage ? 1 : 0);

    // The first frames, and then only when something changes: enough to see the exchange turn over
    // without writing a line per frame forever.
    {
        static int saidFrames = 0;
        static int saidStep = 0;

        if (saidFrames < 8 || saidStep != reached)
        {
            saidFrames++;
            saidStep = reached;
            LOG_DEBUG("DLSS-NR (HIP): tick {} deliver={} stage={} reached step {}", tick, deliver, stage, reached);
        }
    }

    if (reached < 0)
    {
        {
            std::lock_guard<std::mutex> held(g.lock);
            g.stage = Exchange::Stage::Idle;
        }

        Fail("faulted while recording step " + std::to_string(-reached) + " (1 signal, 2-4 deliver, 5-6 stage)");
        return 0;
    }

    std::lock_guard<std::mutex> held(g.lock);
    return g.haveAnswer ? 1 : 0;
}

void Release()
{
    std::lock_guard<std::mutex> once(g_once);

    if (g.worker.joinable())
    {
        {
            std::lock_guard<std::mutex> held(g.lock);
            g.quit = true;
            g.wake.notify_one();
        }

        g.worker.join();
    }

    if (g.readback != nullptr)
    {
        g.readback->Unmap(0, nullptr);
        g.readback->Release();
        g.readback = nullptr;
    }

    if (g.upload != nullptr)
    {
        g.upload->Unmap(0, nullptr);
        g.upload->Release();
        g.upload = nullptr;
    }

    if (g.fence != nullptr)
    {
        g.fence->Release();
        g.fence = nullptr;
    }

    if (g.modelReady && g.shutdown != nullptr)
        g.shutdown();

    if (g.library != nullptr)
    {
        FreeLibrary(g.library);
        g.library = nullptr;
    }

    g.modelReady = false;
    g.stage = Exchange::Stage::Idle;
    g.haveAnswer = false;
    g.quit = false;
    g.readbackMapped = nullptr;
    g.uploadMapped = nullptr;
}

const char* Status()
{
    // A copy per calling thread, so the pointer stays valid while the shared string moves on.
    static thread_local std::string mine;
    {
        std::lock_guard<std::mutex> held(g_statusLock);
        mine = g_status;
    }
    return mine.c_str();
}

float LastModelMs()
{
    std::lock_guard<std::mutex> held(g.lock);
    return g.lastMs;
}

} // namespace Hip
} // namespace DlssNr
