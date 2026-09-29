#include "pch.h"

#include "DlssNr_Hip.h"

#include <Config.h>
#include <Logger.h>
#include <Util.h>

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

// See DlssNr_Hip.h for what this is. In short: the same model, run on the GPU by a Linux process on
// the host, because NGX cannot run it here and ROCm cannot run in here.
//
// The model used to live in this process, loaded through dlss5_hip.dll and a preloaded
// libdlss5_hip.so. It now lives in dlss5-nr-daemon, an ordinary Linux process on the host that holds
// the weights and answers frames on 127.0.0.1, where it is measured at 234 ms a frame on a 7900 XT.
//
// Read the reason for that carefully, because the first version of this comment got it wrong. The
// claim was that a run with the recording switched off still lost the device, so the recording could
// not be the cause and the model had to be what did it. The switch was broken: the queue signal sat
// above its gate, so that run still created both staging buffers, started the worker and signalled
// the game's queue every frame. It proved nothing. What is actually known:
//
//   - with the model out of the process entirely, the game still dies at the same moment, so ROCm
//     in the game process was not the only cause and may not have been a cause at all
//   - with this backend failing at Ensure -- nothing created, nothing recorded -- the game runs
//
// So the culprit is somewhere between those two, in what this file does on the game's device, and
// Mode() below exists to say where. Moving the model out is still right: it removes an unknown and
// it is the only place the network reaches full speed. But it was not the fix.

namespace DlssNr
{
namespace Hip
{
namespace
{

// ---------------------------------------------------------------------------------------------
// The wire.
//
// One frame per exchange: header, pixels, header, pixels. No negotiation and no versions to reconcile
// -- the daemon ships with this DLL and a mismatched magic just refuses the connection. Loopback
// carries the 33 MB of a 1080p frame in a couple of milliseconds, which does not register against the
// model's ~230, so there is nothing to be clever about here.

constexpr uint32_t kRequestMagic = 0x31524E44;  // "DNR1"
constexpr uint32_t kResponseMagic = 0x52524E44; // "DNRR"
constexpr unsigned short kDefaultPort = 47820;

struct Request
{
    uint32_t magic;
    uint32_t width;
    uint32_t height;
    uint32_t format;   // the DXGI format the pixels are in
    uint32_t rowPitch; // bytes per row, as D3D12 laid the staging copy out
    uint32_t bytes;    // rowPitch * height
    uint32_t reset;    // history reset, for when the model grows one
    uint32_t sequence;
};

struct Response
{
    uint32_t magic;
    uint32_t status; // 0 ok
    uint32_t bytes;
    uint32_t millis; // what the model took on the other side
};

// Printed at init. Two builds in a row produced an identical failure, and nothing in the log said
// whether the second one was the DLL actually being loaded.
constexpr const char* kBuildMark = "2026-09-29b";

// ---------------------------------------------------------------------------------------------
// Pixel conversion does not happen here any more.
//
// The staged frame goes to the daemon in its own format, byte for byte as D3D12 laid it out, and
// comes back the same way. Converting there costs nothing extra -- it is the same loop on the same
// CPU -- and it keeps this side down to a socket. All that is left to decide here is whether the
// format is one the other end knows, so an unsupported one fails with a name instead of a picture.

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

// ---------------------------------------------------------------------------------------------
// The exchange.

struct Exchange
{
    // What the model is reached through: one long-lived connection to the daemon, owned by the
    // worker thread and used by nobody else.
    SOCKET link = INVALID_SOCKET;
    bool winsock = false;
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

    // The other clock. Mode 5 never touches the game's queue, so it cannot signal a fence and cannot
    // know when a copy ran; instead it counts frames and assumes a copy recorded three frames ago has
    // long since executed. At four model frames a second the margin is enormous.
    UINT64 frame = 0;
    UINT64 recordedAtFrame = 0;
    UINT64 deliveredAtFrame = 0;

    // Which frame the answer currently in `output` was made FROM -- not when it arrived. The difference
    // is the whole point: the reprojection needs to know how far the world has moved since the model
    // saw it, and that is measured from the frame that was staged, not the frame that took delivery.
    UINT64 answerFromFrame = 0;

    // When the last answer was delivered, and the smoothed gap between deliveries. Distinct from
    // lastMs, which is how long one answer took to make: this is how long each one is on screen.
    std::chrono::steady_clock::time_point lastDelivery {};
    float answerIntervalMs = 0.0f;

    bool quit = false;
    unsigned int seed = 0;
    float lastMs = 0.0f;
    std::thread worker;
};

Exchange g;
std::mutex g_once;       // serialises Ensure/Release against each other
std::mutex g_statusLock; // the worker reports failures too, so this is its own lock
std::string g_status = "not started";
std::atomic<bool> g_failed { false }; // sticky: a failed backend does not retry into a crash

// What the last Evaluate did, for StagedLastFrame/DeliveredLastFrame. Atomics rather than members of
// g because they are read without g.lock by a caller that is already on the render thread -- the same
// thread that wrote them, one call earlier in the same function.
std::atomic<bool> g_stagedLast { false };
std::atomic<bool> g_deliveredLast { false };

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

unsigned short DaemonPort()
{
    char value[16] {};

    if (GetEnvironmentVariableA("DLSS5_NR_PORT", value, sizeof(value)) != 0)
    {
        const int asked = atoi(value);

        if (asked > 0 && asked < 65536)
            return (unsigned short) asked;
    }

    return kDefaultPort;
}

// send and recv are free to do less than they were asked, and a 33 MB frame guarantees they will.
bool SendAll(SOCKET s, const void* from, size_t bytes)
{
    const auto* p = static_cast<const char*>(from);

    while (bytes > 0)
    {
        const int chunk = (int) (bytes > (1 << 20) ? (1 << 20) : bytes);
        const int put = send(s, p, chunk, 0);

        if (put <= 0)
            return false;

        p += put;
        bytes -= size_t(put);
    }

    return true;
}

bool RecvAll(SOCKET s, void* into, size_t bytes)
{
    auto* p = static_cast<char*>(into);

    while (bytes > 0)
    {
        const int chunk = (int) (bytes > (1 << 20) ? (1 << 20) : bytes);
        const int got = recv(s, p, chunk, 0);

        if (got <= 0)
            return false;

        p += got;
        bytes -= size_t(got);
    }

    return true;
}

// Connects to the daemon on loopback. Called once from Ensure, on the thread that is already
// prepared to fail cleanly, so a daemon that is not running costs a message rather than a hang.
SOCKET ConnectToDaemon(unsigned short port, std::string& why)
{
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (s == INVALID_SOCKET)
    {
        why = "no socket";
        return INVALID_SOCKET;
    }

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);

    if (connect(s, (const sockaddr*) &address, sizeof(address)) != 0)
    {
        why = "nothing is listening on 127.0.0.1:" + std::to_string(port);
        closesocket(s);
        return INVALID_SOCKET;
    }

    // Frames are big and latency is what matters, so no Nagle. The timeouts keep a daemon that has
    // died mid-frame from parking the worker thread forever; they are generous because the model
    // itself takes a quarter of a second.
    int nodelay = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &nodelay, sizeof(nodelay));
    int timeout = 15000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*) &timeout, sizeof(timeout));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*) &timeout, sizeof(timeout));
    return s;
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
        // upload buffer last carried. In mode 5 there is no fence at all and Evaluate has already
        // counted the frames, so there is nothing to wait for here.
        if (g.fence != nullptr && ready != nullptr)
        {
            if (g.fence->GetCompletedValue() < waitFor)
            {
                if (SUCCEEDED(g.fence->SetEventOnCompletion(waitFor, ready)))
                    WaitForSingleObject(ready, 5000);
            }
        }

        // The staged frame goes out as it lies in the readback buffer, and the answer comes back in
        // the same layout, straight into the upload buffer the next frame copies from. Neither
        // buffer is touched by the GPU at this point -- that is what the fence wait above was for --
        // and both are persistently mapped, so this is one copy each way and no allocation.
        const auto rowPitch = size_t(g.footprint.Footprint.RowPitch);
        const auto payload = rowPitch * size_t(ModelHeight);

        Request request {};
        request.magic = kRequestMagic;
        request.width = ModelWidth;
        request.height = ModelHeight;
        request.format = (uint32_t) g.format;
        request.rowPitch = (uint32_t) rowPitch;
        request.bytes = (uint32_t) payload;
        request.reset = 0; // the network is stateless today; the field is here for when it is not
        request.sequence = g.seed++;

        const auto started = std::chrono::steady_clock::now();

        if (!SendAll(g.link, &request, sizeof(request)) || !SendAll(g.link, g.readbackMapped, payload))
        {
            std::lock_guard<std::mutex> held(g.lock);
            g.stage = Exchange::Stage::Idle;
            Fail("the connection to dlss5-nr-daemon broke while sending a frame");
            continue;
        }

        Response response {};

        if (!RecvAll(g.link, &response, sizeof(response)) || response.magic != kResponseMagic)
        {
            std::lock_guard<std::mutex> held(g.lock);
            g.stage = Exchange::Stage::Idle;
            Fail("dlss5-nr-daemon stopped answering");
            continue;
        }

        if (response.status != 0 || response.bytes != payload)
        {
            std::lock_guard<std::mutex> held(g.lock);
            g.stage = Exchange::Stage::Idle;
            Fail("the model failed on the daemon side (status " + std::to_string(response.status) +
                 ") -- its console says why");
            continue;
        }

        if (!RecvAll(g.link, g.uploadMapped, payload))
        {
            std::lock_guard<std::mutex> held(g.lock);
            g.stage = Exchange::Stage::Idle;
            Fail("the connection to dlss5-nr-daemon broke while receiving a frame");
            continue;
        }

        const auto ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();

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
                  ID3D12Resource* output, unsigned long long tick, int deliver, int stage, int signalQueue)
{
    int step = 0;

    __try
    {
        step = 1;

        if (signalQueue)
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
            // modelInput arrives as NON_PIXEL_SHADER_RESOURCE, not a UAV. The pass transitions it
            // there before the model is called, because NGX reads the proxy as a shader resource:
            //
            //   Barrier(cmdList, g_nr.colorCopy,  UNORDERED_ACCESS -> NON_PIXEL_SHADER_RESOURCE)
            //   Barrier(cmdList, g_nr.colorSmall, UNORDERED_ACCESS -> NON_PIXEL_SHADER_RESOURCE)
            //
            // and restores it afterwards. A barrier is not validated when it is recorded, only when
            // it runs, so naming the wrong before-state records cleanly and then faults the queue --
            // which is how the first version reached step 6 and lost the device two seconds later.
            step = 5;
            Barrier(cmdList, modelInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyTextureToBuffer(cmdList, modelInput, g.readback);
            Barrier(cmdList, modelInput, D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            step = 6;
        }
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return -step;
    }

    return step;
}

// Printed once, on the first frame that gets this far. A device loss names nothing that caused it, so
// the cheap facts go in the log while they can still be collected: above all whether the command queue
// the pass handed us belongs to the same device as the textures it hands us. Signalling a fence from a
// queue of one device against a fence created on another is exactly the sort of thing that takes a GPU
// down two seconds later, and it would look identical to everything else we have seen.
void Describe(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Resource* output, const D3D12_RESOURCE_DESC& desc,
              int mode)
{
    const void* queueDevice = nullptr;
    D3D12_COMMAND_QUEUE_DESC queueDesc {};

    if (queue != nullptr)
    {
        ID3D12Device* owner = nullptr;

        if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&owner))) && owner != nullptr)
        {
            queueDevice = owner;
            owner->Release();
        }

        queueDesc = queue->GetDesc();
    }

    const char* verdict = queue == nullptr         ? "(no queue was handed to us)"
                          : queueDevice == nullptr ? "(the queue would not name its device)"
                          : queueDevice == device  ? "same device"
                                                   : "DIFFERENT DEVICE -- this alone would explain the loss";

    const D3D12_RESOURCE_DESC out = output->GetDesc();

    LOG_INFO("DLSS-NR (HIP): mode {} | texture device {} | queue device {} -> {} | queue type {} flags {}", mode,
             (const void*) device, queueDevice, verdict, (int) queueDesc.Type, (int) queueDesc.Flags);
    LOG_INFO("DLSS-NR (HIP): input {}x{} fmt {} flags {} | output {}x{} fmt {} flags {}", (unsigned) desc.Width,
             (unsigned) desc.Height, (int) desc.Format, (int) desc.Flags, (unsigned) out.Width, (unsigned) out.Height,
             (int) out.Format, (int) out.Flags);
}

bool CreateStaging(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, bool needFence)
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

    if (needFence && FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence))))
    {
        Fail("the fence could not be created");
        return false;
    }

    // Named for the same reason the pass's textures are: when a hang report says a use-after-free
    // happened at some address, "DLSS-NR readback" answers in one line what a cookie number cannot.
    g.readback->SetName(L"DLSS-NR readback");
    g.upload->SetName(L"DLSS-NR upload");

    g.stagedBytes = total;
    g.format = desc.Format;
    return true;
}

// How many frames to let pass before assuming a recorded copy has run, in the modes that have no fence to
// ask -- at both ends of the exchange. DLSS5_NR_FRAME_LAG=1..3; unset is 3, the original constant.
//
// Three was chosen when the model took ~230 ms and six frames of waiting per cycle cost nothing. At 84 FPS
// they are ~71 ms of a ~228 ms cycle, and all of it ages the answer. Shorter is only safe while the copy
// recorded that many frames ago has really executed: too short and the model occasionally reads a
// half-written frame, which shows up as a one-answer glitch rather than as a steady fault.
UINT64 FrameLag()
{
    static int lag = 0;

    if (lag == 0)
    {
        char value[16] {};
        lag = 3;

        if (GetEnvironmentVariableA("DLSS5_NR_FRAME_LAG", value, sizeof(value)) != 0)
        {
            const int asked = atoi(value);

            if (asked >= 1 && asked <= 3)
                lag = asked;
        }

        LOG_INFO("DLSS-NR (HIP): frame lag {} at each end of the exchange{}", lag,
                 lag == 3 ? "" : " (DLSS5_NR_FRAME_LAG)");
    }

    return UINT64(lag);
}

// The gap between deliveries, smoothed. Called from Evaluate with g.lock held.
//
// Recorded on every delivery in both clock regimes, so the number does not silently become zero if
// the queue-driven modes are ever used again.
void NoteDeliveryLocked()
{
    const auto now = std::chrono::steady_clock::now();

    if (g.lastDelivery.time_since_epoch().count() != 0)
    {
        const float gap = std::chrono::duration<float, std::milli>(now - g.lastDelivery).count();

        // A gap of seconds is a load screen, an alt-tab or a menu, not a delivery interval. Rejected
        // rather than smoothed in, for the same reason the frame-time average rejects its outliers:
        // one of them takes a hundred deliveries to decay back out.
        if (gap > 1.0f && gap < 2000.0f)
            g.answerIntervalMs =
                g.answerIntervalMs == 0.0f ? gap : g.answerIntervalMs * 0.8f + gap * 0.2f;
    }

    g.lastDelivery = now;

    // What the cycle actually costs, logged often enough to read off a short test and rarely enough not to
    // flood the file. The interval is the number DLSS5_NR_FRAME_LAG should move: half of it is added to
    // every answer's age on screen.
    static unsigned int delivered = 0;

    if (++delivered % 32 == 0 && g.answerIntervalMs > 0.0f)
        LOG_INFO("DLSS-NR (HIP): answer every {:.1f} ms (model round trip {:.1f} ms, frame lag {})",
                 g.answerIntervalMs, g.lastMs, (unsigned int) FrameLag());
}

// ---------------------------------------------------------------------------------------------
// Per-frame synchronous mode: DLSS5_NR_SYNC=1 (0030).
//
// The exchange above runs one answer at a time and lets the game run ahead of it, so what is on
// screen is an answer many frames old and the fade, the accumulation and the delta exist to hide
// that. This mode does the opposite: EVERY frame is sent to the model, and the render thread waits
// for the answer, so the game runs at the model's pace and what is on screen is exactly one frame
// old. One frame, not zero, because the answer cannot be put back into the list that is still being
// recorded; it goes into the next one:
//
//   Evaluate(N), recording the game's list L(N):
//     1. wait -- on this thread, with a timeout -- for the answer to frame N-1
//     2. record: copy that answer (upload slot) over the output; the resolve composes it
//     3. record: stage proxy N into readback slot N % 3, then copy N into that slot's flag
//     4. queue job N for the worker
//   worker: watch the flag until it reads N -- the game's GPU has now executed L(N)'s copy --
//           send slot N % 3 to the daemon, receive the answer into upload slot N % 3, mark done
//
// Nothing here touches the game's queue: the flag replaces both the fence of modes 2-4 and the
// frame counting of mode 5, and it fires the moment the copy has run instead of three frames later.
//
// Why three slots are enough: an answer's upload slot is next written by job N+3, which starts only
// after its flag -- i.e. after L(N+3), and so L(N+1) that read the answer, has executed. A readback
// slot is next written by frame N+3's staging, which is recorded only after the render thread has
// waited for answer N+2, and the worker takes jobs in order.
//
// A game that records frame N+1 before submitting frame N would make step 1 wait for a copy that
// cannot run yet. The wait therefore has a timeout (DLSS5_NR_SYNC_WAIT_MS, 400): on expiry the frame
// is composed with the answer it already has and staging goes on, so such a game degrades instead of
// hanging. The log counts those.

bool SyncRequested()
{
    static int on = -1;

    if (on < 0)
    {
        char value[16] {};
        on = GetEnvironmentVariableA("DLSS5_NR_SYNC", value, sizeof(value)) != 0 && value[0] == '1' &&
                     value[1] == 0
                 ? 1
                 : 0;
    }

    return on == 1;
}

// Sync needs mode 5's rules -- deliver, stage, never the queue. Any other mode keeps its own exchange.
bool SyncActive(int mode)
{
    if (!SyncRequested())
        return false;

    if (mode != 5)
    {
        static bool said = false;

        if (!said)
        {
            said = true;
            LOG_WARN("DLSS-NR (HIP): DLSS5_NR_SYNC=1 needs DLSS5_NR_MODE 5 (the default); mode {} keeps the "
                     "asynchronous exchange",
                     mode);
        }

        return false;
    }

    return true;
}

std::chrono::milliseconds SyncWait()
{
    static int ms = -1;

    if (ms < 0)
    {
        char value[16] {};
        ms = 400;

        if (GetEnvironmentVariableA("DLSS5_NR_SYNC_WAIT_MS", value, sizeof(value)) != 0)
        {
            const int asked = atoi(value);

            if (asked >= 10 && asked <= 5000)
                ms = asked;
        }
    }

    return std::chrono::milliseconds(ms);
}

struct SyncExchange
{
    static constexpr unsigned int kSlots = 3;
    static constexpr unsigned int kSeqRing = 64;
    static constexpr UINT64 kStride = 256; // one flag / one ring entry per 256 bytes

    // Slot 0 is the exchange's own readback/upload pair; 1 and 2 are made here.
    ID3D12Resource* readback[kSlots] {};
    ID3D12Resource* upload[kSlots] {};
    uint8_t* readbackMapped[kSlots] {};
    uint8_t* uploadMapped[kSlots] {};

    // The flags the worker watches (READBACK), the frame numbers the list copies into them (an UPLOAD
    // ring written at record time), and a DEFAULT hop between the two -- see GuardedRecordSync.
    ID3D12Resource* flags = nullptr;
    const volatile uint64_t* flagsMapped = nullptr;
    ID3D12Resource* seq = nullptr;
    uint64_t* seqMapped = nullptr;
    ID3D12Resource* seqHop = nullptr;

    enum class Slot
    {
        Free,
        Queued,
        Running,
        Done,
        Failed
    };

    Slot state[kSlots] { Slot::Free, Slot::Free, Slot::Free };
    UINT64 frameOf[kSlots] {};
    std::condition_variable done; // with g.lock

    // For the log.
    float waitMs = 0.0f;  // the render thread, waiting for the previous answer
    float readyMs = 0.0f; // the worker, waiting for the game's GPU to run the staging copy
    unsigned int timeouts = 0;
    unsigned int skipped = 0;
};

SyncExchange gs;

bool CreateSyncStaging(ID3D12Device* device)
{
    D3D12_RESOURCE_DESC bufferDesc {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = g.stagedBytes;
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
    D3D12_HEAP_PROPERTIES defaultHeap {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RANGE nothing { 0, 0 };

    gs.readback[0] = g.readback;
    gs.upload[0] = g.upload;
    gs.readbackMapped[0] = g.readbackMapped;
    gs.uploadMapped[0] = g.uploadMapped;

    for (unsigned int i = 1; i < SyncExchange::kSlots; ++i)
    {
        if (FAILED(device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&gs.readback[i]))) ||
            FAILED(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                   IID_PPV_ARGS(&gs.upload[i]))) ||
            FAILED(gs.readback[i]->Map(0, nullptr, reinterpret_cast<void**>(&gs.readbackMapped[i]))) ||
            FAILED(gs.upload[i]->Map(0, &nothing, reinterpret_cast<void**>(&gs.uploadMapped[i]))))
        {
            Fail("the per-frame staging slots could not be allocated");
            return false;
        }

        gs.readback[i]->SetName(i == 1 ? L"DLSS-NR readback 1" : L"DLSS-NR readback 2");
        gs.upload[i]->SetName(i == 1 ? L"DLSS-NR upload 1" : L"DLSS-NR upload 2");
    }

    D3D12_RESOURCE_DESC flagDesc = bufferDesc;
    flagDesc.Width = SyncExchange::kStride * SyncExchange::kSlots;
    D3D12_RESOURCE_DESC ringDesc = bufferDesc;
    ringDesc.Width = SyncExchange::kStride * SyncExchange::kSeqRing;
    D3D12_RESOURCE_DESC hopDesc = bufferDesc;
    hopDesc.Width = SyncExchange::kStride;

    uint64_t* flagsWritable = nullptr;

    if (FAILED(device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &flagDesc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gs.flags))) ||
        FAILED(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &ringDesc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&gs.seq))) ||
        FAILED(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &hopDesc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gs.seqHop))) ||
        FAILED(gs.flags->Map(0, nullptr, reinterpret_cast<void**>(&flagsWritable))) ||
        FAILED(gs.seq->Map(0, &nothing, reinterpret_cast<void**>(&gs.seqMapped))))
    {
        Fail("the per-frame flags could not be allocated");
        return false;
    }

    // Frame numbers start at 1, so a zeroed flag can never be mistaken for a staged frame.
    for (unsigned int i = 0; i < SyncExchange::kSlots; ++i)
        flagsWritable[i * (SyncExchange::kStride / 8)] = 0;

    gs.flagsMapped = flagsWritable;
    gs.flags->SetName(L"DLSS-NR sync flags");
    gs.seq->SetName(L"DLSS-NR sync ring");
    gs.seqHop->SetName(L"DLSS-NR sync hop");
    return true;
}

void ReleaseSyncStaging()
{
    for (unsigned int i = 1; i < SyncExchange::kSlots; ++i)
    {
        if (gs.readback[i] != nullptr)
        {
            gs.readback[i]->Unmap(0, nullptr);
            gs.readback[i]->Release();
        }

        if (gs.upload[i] != nullptr)
        {
            gs.upload[i]->Unmap(0, nullptr);
            gs.upload[i]->Release();
        }
    }

    if (gs.flags != nullptr)
    {
        gs.flags->Unmap(0, nullptr);
        gs.flags->Release();
    }

    if (gs.seq != nullptr)
    {
        gs.seq->Unmap(0, nullptr);
        gs.seq->Release();
    }

    if (gs.seqHop != nullptr)
        gs.seqHop->Release();

    // Slot 0 belongs to the exchange, which releases it itself.
    for (unsigned int i = 0; i < SyncExchange::kSlots; ++i)
    {
        gs.readback[i] = nullptr;
        gs.upload[i] = nullptr;
        gs.readbackMapped[i] = nullptr;
        gs.uploadMapped[i] = nullptr;
        gs.state[i] = SyncExchange::Slot::Free;
        gs.frameOf[i] = 0;
    }

    gs.flags = nullptr;
    gs.flagsMapped = nullptr;
    gs.seq = nullptr;
    gs.seqMapped = nullptr;
    gs.seqHop = nullptr;
}

void WorkerSyncMain()
{
    using Slot = SyncExchange::Slot;

    for (;;)
    {
        unsigned int slot = 0;
        UINT64 frame = 0;

        {
            std::unique_lock<std::mutex> held(g.lock);
            g.wake.wait(held,
                        []
                        {
                            if (g.quit)
                                return true;

                            for (auto st : gs.state)
                                if (st == Slot::Queued)
                                    return true;

                            return false;
                        });

            if (g.quit)
                break;

            // Oldest first: answers are delivered in frame order.
            frame = ~UINT64(0);

            for (unsigned int i = 0; i < SyncExchange::kSlots; ++i)
            {
                if (gs.state[i] == Slot::Queued && gs.frameOf[i] < frame)
                {
                    frame = gs.frameOf[i];
                    slot = i;
                }
            }

            gs.state[slot] = Slot::Running;
        }

        // Wait for the game's GPU to execute the list that staged this frame. Spin briefly, then sleep
        // a millisecond at a time; give up after two seconds, which only a list that was never
        // submitted can take.
        const auto started = std::chrono::steady_clock::now();
        bool ready = false;

        for (;;)
        {
            if (gs.flagsMapped[slot * (SyncExchange::kStride / 8)] == frame)
            {
                ready = true;
                break;
            }

            const auto waited = std::chrono::steady_clock::now() - started;

            if (waited > std::chrono::seconds(2))
                break;

            {
                std::lock_guard<std::mutex> held(g.lock);

                if (g.quit)
                    break;
            }

            if (waited < std::chrono::milliseconds(20))
                std::this_thread::yield();
            else
                Sleep(1);
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        if (!ready)
        {
            std::lock_guard<std::mutex> held(g.lock);
            gs.state[slot] = Slot::Failed;
            gs.done.notify_all();
            continue;
        }

        const auto copied = std::chrono::steady_clock::now();
        const auto rowPitch = size_t(g.footprint.Footprint.RowPitch);
        const auto payload = rowPitch * size_t(ModelHeight);

        Request request {};
        request.magic = kRequestMagic;
        request.width = ModelWidth;
        request.height = ModelHeight;
        request.format = (uint32_t) g.format;
        request.rowPitch = (uint32_t) rowPitch;
        request.bytes = (uint32_t) payload;
        request.reset = 0;
        request.sequence = (uint32_t) frame;

        Response response {};
        const char* broke = nullptr;

        if (!SendAll(g.link, &request, sizeof(request)) || !SendAll(g.link, gs.readbackMapped[slot], payload))
            broke = "the connection to dlss5-nr-daemon broke while sending a frame";
        else if (!RecvAll(g.link, &response, sizeof(response)) || response.magic != kResponseMagic)
            broke = "dlss5-nr-daemon stopped answering";
        else if (response.status != 0 || response.bytes != payload)
            broke = "the model failed on the daemon side -- its console says why";
        else if (!RecvAll(g.link, gs.uploadMapped[slot], payload))
            broke = "the connection to dlss5-nr-daemon broke while receiving a frame";

        const auto finished = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> held(g.lock);
            gs.state[slot] = broke != nullptr ? Slot::Failed : Slot::Done;
            gs.readyMs = std::chrono::duration<float, std::milli>(copied - started).count();

            if (broke == nullptr)
                g.lastMs = std::chrono::duration<float, std::milli>(finished - copied).count();

            gs.done.notify_all();
        }

        if (broke != nullptr)
            Fail(broke);
    }
}

// The per-frame counterpart of GuardedRecord: every call on the game's list, no C++ object in scope.
//
// The flag must not land before the staged pixels have. Two copies in one list are not ordered unless
// a barrier says so, so the frame number takes a hop through a DEFAULT buffer whose transition between
// the two copies (COPY_DEST -> COPY_SOURCE, a transfer-to-transfer dependency) orders the staging copy
// before the flag copy; a global UAV barrier sits beside it for runtimes that elide buffer transitions.
//
// Returns the step reached (2-4 deliver, 5-8 stage); negative means it faulted there.
int GuardedRecordSync(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* modelInput, ID3D12Resource* output,
                      ID3D12Resource* deliverFrom, ID3D12Resource* stageTo, UINT64 flagOffset, UINT64 seqOffset)
{
    int step = 0;

    __try
    {
        if (deliverFrom != nullptr)
        {
            step = 2;
            Barrier(cmdList, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            step = 3;
            CopyBufferToTexture(cmdList, deliverFrom, output);
            step = 4;
            Barrier(cmdList, output, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        if (stageTo != nullptr)
        {
            step = 5;
            cmdList->CopyBufferRegion(gs.seqHop, 0, gs.seq, seqOffset, 8);

            step = 6;
            Barrier(cmdList, modelInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
            CopyTextureToBuffer(cmdList, modelInput, stageTo);
            Barrier(cmdList, modelInput, D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            step = 7;
            Barrier(cmdList, gs.seqHop, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_RESOURCE_BARRIER all {};
            all.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            all.UAV.pResource = nullptr;
            cmdList->ResourceBarrier(1, &all);
            cmdList->CopyBufferRegion(gs.flags, flagOffset, gs.seqHop, 0, 8);
            Barrier(cmdList, gs.seqHop, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            step = 8;
        }
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return -step;
    }

    return step;
}

int EvaluateSync(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* modelInput, ID3D12Resource* output, bool reset)
{
    using Slot = SyncExchange::Slot;
    const UINT64 frame = ++g.frame;
    int deliverSlot = -1;
    int stageSlot = -1;

    {
        std::unique_lock<std::mutex> held(g.lock);

        if (reset)
            g.haveAnswer = false;

        // 1. The previous frame's answer. The render thread waits here, and that wait is the point of
        //    the mode: it is what makes the game run at the model's pace.
        int previous = -1;

        for (unsigned int i = 0; i < SyncExchange::kSlots; ++i)
            if (gs.frameOf[i] == frame - 1 && gs.state[i] != Slot::Free)
                previous = (int) i;

        const auto waitStart = std::chrono::steady_clock::now();

        if (previous >= 0)
        {
            gs.done.wait_until(held, waitStart + SyncWait(),
                               [previous]
                               {
                                   return g.quit || gs.state[previous] == Slot::Done ||
                                          gs.state[previous] == Slot::Failed;
                               });

            if (gs.state[previous] == Slot::Done)
            {
                deliverSlot = previous;
                gs.state[previous] = Slot::Free;
                g.haveAnswer = true;
                g.deliveredAtFrame = frame;
                g.answerFromFrame = frame - 1;
                NoteDeliveryLocked();
            }
            else if (gs.state[previous] == Slot::Failed)
            {
                gs.state[previous] = Slot::Free;
            }
            else
            {
                gs.timeouts++; // still in flight: compose with the answer already on screen
            }
        }

        gs.waitMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - waitStart).count();

        // 2. This frame. A slot still Done holds an answer that missed its frame; it is simply reused.
        const unsigned int slot = (unsigned int) (frame % SyncExchange::kSlots);

        if (gs.state[slot] == Slot::Free || gs.state[slot] == Slot::Done || gs.state[slot] == Slot::Failed)
            stageSlot = (int) slot;
        else
            gs.skipped++;
    }

    const UINT64 seqOffset = (frame % SyncExchange::kSeqRing) * SyncExchange::kStride;

    if (stageSlot >= 0)
        gs.seqMapped[seqOffset / 8] = frame;

    const int reached =
        GuardedRecordSync(cmdList, modelInput, output, deliverSlot >= 0 ? gs.upload[deliverSlot] : nullptr,
                          stageSlot >= 0 ? gs.readback[stageSlot] : nullptr,
                          UINT64(stageSlot >= 0 ? stageSlot : 0) * SyncExchange::kStride, seqOffset);

    if (reached < 0)
    {
        Fail("faulted while recording per-frame step " + std::to_string(-reached) + " (2-4 deliver, 5-8 stage)");
        return 0;
    }

    bool haveAnswer = false;

    {
        std::lock_guard<std::mutex> held(g.lock);

        if (stageSlot >= 0)
        {
            gs.frameOf[stageSlot] = frame;
            gs.state[stageSlot] = Slot::Queued;
            g.recordedAtFrame = frame;
            g.wake.notify_one();
        }

        haveAnswer = g.haveAnswer;

        // What each frame costs, every 64 frames: how long the render thread waited, how long the
        // game's GPU took to reach the staging copy, and the model's round trip.
        if (frame % 64 == 0)
            LOG_INFO("DLSS-NR (HIP sync): frame {} | render thread waited {:.1f} ms | staging copy ran after "
                     "{:.1f} ms | model round trip {:.1f} ms | answer every {:.1f} ms | timeouts {} | skipped {}",
                     frame, gs.waitMs, gs.readyMs, g.lastMs, g.answerIntervalMs, gs.timeouts, gs.skipped);
    }

    g_stagedLast = stageSlot >= 0;
    g_deliveredLast = deliverSlot >= 0;
    return haveAnswer ? 1 : 0;
}

} // namespace

// What this backend is allowed to do, from DLSS5_NR_MODE. Each mode is a strict superset of the one
// below it, and mode 0 really does nothing:
//
//   0  nothing at all -- Evaluate returns immediately; the pass runs on without a model
//   1  create the staging buffers, map them, start the worker, and stop there
//   2  as 1, plus signal the game's command queue once per frame
//   3  as 2, plus record the staging copy (the proxy going out)
//   4  as 3, plus deliver the answer back over the output -- this is what has been crashing
//   5  as 4 but never touching the game's queue: no fence, no Signal. Copies go on the game's list
//      and frames are counted instead. The default, because it is the one thing not yet ruled out.
//
// The switch this replaces was wrong in a way that cost a night. The queue signal sat *above* the
// gate, so mode "0" still created every resource, started the worker and signalled the game's queue
// on every frame -- and its crash was read as proof that nothing recorded here could matter. It was
// not proof of anything. Hence the nesting above, and hence mode 0 returning before it touches
// anything at all.
int Mode()
{
    static int mode = -1;

    if (mode < 0)
    {
        char value[16] {};
        mode = 5;

        if (GetEnvironmentVariableA("DLSS5_NR_MODE", value, sizeof(value)) != 0)
        {
            const int asked = atoi(value);

            if (asked >= 0 && asked <= 5)
                mode = asked;
        }
    }

    return mode;
}

// The frame lag is FrameLag(), above: adjustable with DLSS5_NR_FRAME_LAG, 3 when unset.

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

// ---------------------------------------------------------------------------------------------
// In-process probe (stage 2 of the in-game path; DLSS5_NR_INPROC_PROBE=1, 2026-09-29).
//
// Answers one question before any in-process transport is written: does the model come up and run
// INSIDE this game process? It loads dlss5_hip.dll from OptiScaler's folder (the PE trampoline, which
// finds the preloaded libdlss5_hip_preload.so through the live Unix environment), initialises the
// model, runs it five times on a fixed synthetic 1920x1080 frame, logs the times and an output digest,
// and shuts it down again. The daemon path is untouched and keeps running the game's frames.
//
// hip/tests/probe_inproc_host.py runs the same frame on the host and prints the same digest: equal
// digests mean the in-process model computes exactly what the daemon's does.
namespace InProcessProbe
{
using InitFn = int (*)(const char*, int);
using RunFn = int (*)(const float*, float*, unsigned int);
using ErrorFn = const char* (*)();
using ShutdownFn = void (*)();

constexpr uint32_t kWidth = 1920;
constexpr uint32_t kHeight = 1080;

bool Requested()
{
    char value[8] {};
    return GetEnvironmentVariableA("DLSS5_NR_INPROC_PROBE", value, sizeof(value)) != 0 && value[0] == '1';
}

std::wstring ModuleFolder()
{
    static const int anchor = 0;
    HMODULE self = nullptr;
    wchar_t path[MAX_PATH] {};

    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&anchor), &self) ||
        GetModuleFileNameW(self, path, MAX_PATH) == 0)
        return L".";

    std::wstring folder(path);
    const size_t slash = folder.find_last_of(L"\\/");

    if (slash != std::wstring::npos)
        folder.resize(slash);

    return folder;
}

// sum over i of bits[i] * (2i + 1), mod 2^64 -- probe_inproc_host.py computes the same.
uint64_t Digest(const std::vector<float>& values)
{
    uint64_t sum = 0;

    for (size_t i = 0; i < values.size(); i++)
    {
        uint32_t bits;
        std::memcpy(&bits, &values[i], sizeof(bits));
        sum += uint64_t(bits) * (uint64_t(i) * 2u + 1u);
    }

    return sum;
}

void Run()
{
    const std::wstring path = ModuleFolder() + L"\\dlss5_hip.dll";
    HMODULE dll = LoadLibraryW(path.c_str());

    if (dll == nullptr)
    {
        LOG_ERROR("DLSS-NR in-process probe: dlss5_hip.dll not loadable from OptiScaler's folder (error {})",
                  (unsigned) GetLastError());
        return;
    }

    auto init = reinterpret_cast<InitFn>(GetProcAddress(dll, "dlss5_init"));
    auto run = reinterpret_cast<RunFn>(GetProcAddress(dll, "dlss5_run"));
    auto lastError = reinterpret_cast<ErrorFn>(GetProcAddress(dll, "dlss5_last_error"));
    auto shutdown = reinterpret_cast<ShutdownFn>(GetProcAddress(dll, "dlss5_shutdown"));

    if (init == nullptr || run == nullptr || lastError == nullptr || shutdown == nullptr)
    {
        LOG_ERROR("DLSS-NR in-process probe: dlss5_hip.dll lacks the dlss5_* exports");
        return;
    }

    auto errorText = [&]()
    {
        const char* text = lastError();
        return std::string(text != nullptr && text[0] != 0 ? text : "(no message)");
    };

    auto started = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point from)
    { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count(); };

    // Null weights: the Linux side reads DLSS5_HIP_WEIGHTS.
    if (init(nullptr, 0) != 0)
    {
        LOG_ERROR("DLSS-NR in-process probe: dlss5_init failed after {:.0f} ms: {}", since(started),
                  errorText());
        return;
    }

    LOG_INFO("DLSS-NR in-process probe: model initialised inside the game in {:.0f} ms", since(started));

    std::vector<float> input(size_t(kWidth) * kHeight * 4);
    std::vector<float> output(size_t(kWidth) * kHeight * 3);

    for (uint32_t y = 0; y < kHeight; y++)
        for (uint32_t x = 0; x < kWidth; x++)
            for (uint32_t c = 0; c < 4; c++)
                input[(size_t(y) * kWidth + x) * 4 + c] = float((x * 7 + y * 13 + c * 29) % 256) * 0.00390625f; // k * 2^-8: exact under /fp:fast

    const uint64_t inputDigest = Digest(input);
    double millis[5] {};
    uint64_t digests[5] {};

    for (int i = 0; i < 5; i++)
    {
        started = std::chrono::steady_clock::now();

        if (run(input.data(), output.data(), 0) != 0)
        {
            LOG_ERROR("DLSS-NR in-process probe: dlss5_run {} failed: {}", i + 1, errorText());
            shutdown();
            return;
        }

        millis[i] = since(started);
        digests[i] = Digest(output);
    }

    bool replay = true;

    for (int i = 1; i < 5; i++)
        replay = replay && digests[i] == digests[0];

    LOG_INFO("DLSS-NR in-process probe: 5 runs {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} ms (wall, incl. host copies) | "
             "input digest {:016x} | output digest {:016x} | replay {}",
             millis[0], millis[1], millis[2], millis[3], millis[4], inputDigest, digests[0], replay ? "equal" : "DIFFERS");

    // The daemon keeps serving the game; free this copy's VRAM.
    shutdown();
    LOG_INFO("DLSS-NR in-process probe: done, model shut down");
}

void StartOnce()
{
    static std::atomic<bool> started { false };

    if (!Requested() || started.exchange(true))
        return;

    LOG_INFO("DLSS-NR in-process probe: starting (DLSS5_NR_INPROC_PROBE=1)");
    std::thread(Run).detach();
}
} // namespace InProcessProbe

bool Ensure(ID3D12Device* device)
{
    std::lock_guard<std::mutex> once(g_once);

    if (g_failed)
        return false;

    if (g.modelReady)
        return true;

    if (device == nullptr)
        return false;

    InProcessProbe::StartOnce();

    // Winsock inside a game process: WSAStartup is reference-counted, and the game has certainly
    // called it already, so this is a formality that keeps the accounting straight.
    if (!g.winsock)
    {
        WSADATA wsa {};

        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        {
            Fail("Winsock would not start");
            return false;
        }

        g.winsock = true;
    }

    const unsigned short port = DaemonPort();
    std::string why;
    g.link = ConnectToDaemon(port, why);

    if (g.link == INVALID_SOCKET)
    {
        Fail("cannot reach dlss5-nr-daemon: " + why +
             " -- start it on the host (dlss5-nr-daemon --weights <dir>) before launching the game");
        return false;
    }

    g.device = device;
    g.modelReady = true;
    Say("ready");
    LOG_INFO("DLSS-NR (HIP): connected to dlss5-nr-daemon on 127.0.0.1:{}, running at {}x{} "
             "[build {} mode {}]",
             port, ModelWidth, ModelHeight, kBuildMark, Mode());
    return true;
}

int Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* queue, ID3D12Resource* modelInput,
             ID3D12Resource* output, unsigned int workWidth, unsigned int workHeight, bool reset)
{
    // Cleared here rather than at each return: Evaluate leaves by eight different paths and a caller
    // reading "staged" from two frames ago would zero its accumulator in the wrong place.
    g_stagedLast = false;
    g_deliveredLast = false;

    if (g_failed || cmdList == nullptr || modelInput == nullptr || output == nullptr)
        return 0;

    const int mode = Mode();

    if (mode == 0)
        return 0;

    if (workWidth != ModelWidth || workHeight != ModelHeight)
    {
        Fail("the model runs at 1920x1080 only, and this frame asked for another size");
        return 0;
    }

    // Modes 2 to 4 signal the game's queue; 5 never speaks to it, so it does not care whether the
    // pass had one to give us.
    const bool useQueue = mode >= 2 && mode <= 4;

    if (useQueue)
    {
        if (queue == nullptr)
        {
            // Without the queue there is no way to know when a copy has run, and reading a buffer the
            // GPU may not have written yet would show the model a frame of noise.
            Say("waiting for the command queue");
            return 0;
        }

        // The fallback queue arrives as a void* OptiScaler keeps for its timing; ask it whether it
        // really is a command queue before calling one. Signal on something that is not would jump
        // through a vtable that isn't there, on the game's render thread, on the first frame.
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

        Describe(device, queue, output, desc, mode);
        const bool sync = SyncActive(mode);
        const bool made = CreateStaging(device, desc, useQueue) && (!sync || CreateSyncStaging(device));
        device->Release();

        if (!made)
            return 0;

        g.worker = std::thread(sync ? WorkerSyncMain : WorkerMain);

        if (sync)
            LOG_INFO("DLSS-NR (HIP): per-frame mode (DLSS5_NR_SYNC=1): every frame goes to the model and the "
                     "render thread waits up to {} ms for the previous frame's answer",
                     (int) SyncWait().count());
    }

    if (mode == 1)
        return 0; // the buffers exist, the worker is up, and nothing else happens. That is the test.

    if (SyncActive(mode))
        return EvaluateSync(cmdList, modelInput, output, reset);

    UINT64 tick = 0;

    if (useQueue)
        tick = ++g.tick; // one tick per frame, ordered after everything already submitted

    const UINT64 frame = ++g.frame;

    if (reset)
    {
        std::lock_guard<std::mutex> held(g.lock);
        g.haveAnswer = false;
    }

    const bool mayStage = mode >= 3;
    const bool mayDeliver = mode >= 4;
    bool stage = false;
    bool deliver = false;

    {
        std::lock_guard<std::mutex> held(g.lock);

        if (useQueue)
        {
            // A copy recorded last frame is on a list that has now been submitted, so this tick is
            // when it will have run.
            if (g.stage == Exchange::Stage::Recorded)
            {
                g.stage = Exchange::Stage::Submitted;
                g.readableAt = tick;
                g.wake.notify_one();
            }
            else if (g.stage == Exchange::Stage::Ready && mayDeliver)
            {
                deliver = true;
                g.stage = Exchange::Stage::Idle;
                g.writableAt = tick + 1; // the copy below runs no later than the next tick
                g.haveAnswer = true;
                NoteDeliveryLocked();
            }
            else if (g.stage == Exchange::Stage::Idle && mayStage)
            {
                stage = true;
                g.stage = Exchange::Stage::Recorded;
            }
        }
        else
        {
            // No fence to ask, so the frame counter answers instead: a copy recorded three frames ago
            // has run, and a buffer the GPU was reading three frames ago is free.
            if (g.stage == Exchange::Stage::Recorded && frame >= g.recordedAtFrame + FrameLag())
            {
                g.stage = Exchange::Stage::Submitted;
                g.wake.notify_one();
            }
            else if (g.stage == Exchange::Stage::Ready && mayDeliver)
            {
                deliver = true;
                g.stage = Exchange::Stage::Idle;
                g.haveAnswer = true;
                g.deliveredAtFrame = frame;

                // The state machine is strictly serial, so nothing has been staged since this answer's
                // own frame was: recordedAtFrame still names it.
                g.answerFromFrame = g.recordedAtFrame;
                NoteDeliveryLocked();
            }
            else if (g.stage == Exchange::Stage::Idle && mayStage &&
                     (g.deliveredAtFrame == 0 || frame >= g.deliveredAtFrame + FrameLag()))
            {
                stage = true;
                g.stage = Exchange::Stage::Recorded;
                g.recordedAtFrame = frame;
            }
        }
    }

    const int reached =
        GuardedRecord(cmdList, queue, modelInput, output, tick, deliver ? 1 : 0, stage ? 1 : 0, useQueue ? 1 : 0);

    // The first frames, and then only when something changes: enough to see the exchange turn over
    // without writing a line per frame forever.
    {
        static int saidFrames = 0;
        static int saidStep = 0;

        if (saidFrames < 8 || saidStep != reached)
        {
            saidFrames++;
            saidStep = reached;
            LOG_DEBUG("DLSS-NR (HIP): mode {} frame {} tick {} deliver={} stage={} reached step {}", mode, frame, tick,
                      deliver, stage, reached);
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

    g_stagedLast = stage;
    g_deliveredLast = deliver;

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

    // Closing the connection is all the daemon needs: it goes back to waiting for the next game and
    // keeps the weights loaded, so a relaunch does not pay the model's start-up again.
    if (g.link != INVALID_SOCKET)
    {
        shutdown(g.link, SD_BOTH);
        closesocket(g.link);
        g.link = INVALID_SOCKET;
    }

    if (g.winsock)
    {
        WSACleanup();
        g.winsock = false;
    }

    ReleaseSyncStaging();

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

// The only thing that means "stop asking". Everything else Evaluate can answer is a frame that had
// nothing to compose, which is not a fault and must not be latched.
bool Failed() { return g_failed; }

bool SyncMode() { return Enabled() && SyncActive(Mode()); }

// How many frames have passed since the scene the current answer describes.
//
// Zero when there is no answer, and zero on the queue-driven modes, which never tracked the staged
// frame -- mode 5 is the default and the only one in use, so that is a gap rather than a bug, but it is
// a gap: a caller that gets 0 warps by nothing and composes exactly as it always did.
unsigned int FramesSinceAnswer()
{
    std::lock_guard<std::mutex> held(g.lock);

    if (!g.haveAnswer || g.answerFromFrame == 0 || g.frame <= g.answerFromFrame)
        return 0;

    return (unsigned int) (g.frame - g.answerFromFrame);
}

float LastModelMs()
{
    std::lock_guard<std::mutex> held(g.lock);
    return g.lastMs;
}

bool StagedLastFrame() { return g_stagedLast.load(); }

bool DeliveredLastFrame() { return g_deliveredLast.load(); }

float LastAnswerIntervalMs()
{
    std::lock_guard<std::mutex> held(g.lock);
    return g.answerIntervalMs;
}

} // namespace Hip
} // namespace DlssNr
