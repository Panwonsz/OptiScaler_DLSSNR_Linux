#pragma once

#include <d3d12.h>

// DLSS 5 Neural Rendering on AMD, through a native HIP implementation of the model.
//
// This is a second backend for the one call in this module that is NVIDIA-only. Everything else the
// pass does -- building the display-referred proxy, the exposure work, the HUD mask, the resolve that
// composes the answer back, capture, frame hold, the menu -- is API-agnostic and runs unchanged. Only
// the model itself cannot run through NGX on a Radeon, and that is the piece replaced here.
//
// The model is the same network, re-implemented in HIP for RDNA 3/4, and it runs outside the game:
//
//   dlss5-nr-daemon   an ordinary Linux process on the host that loads the weights once and answers
//                     frames on 127.0.0.1; started by dlss5-nr-run.sh before the game
//   libdlss5_hip.so   the network itself, which that daemon loads -- with the system's own ROCm,
//                     since nothing has to be smuggled into the Steam container any more
//
// It used to live in this process, loaded through a PE trampoline over a preloaded libdlss5_hip.so.
// That cost us the GPU: Stellar Blade hit VK_ERROR_DEVICE_LOST two seconds into the first frame even
// with the recording switched off entirely (DLSS5_NR_STEPS=0), so the model was doing it merely by
// existing in the process. ROCm compute and vkd3d graphics do not share a Wine process well.
//
// The contract is unchanged and still deliberately narrow: one frame in, one frame out, 1920x1080,
// which is the size the network was built for. The caller therefore has to drive the pass at exactly
// that working size -- see ModelWidth/ModelHeight. What crosses the wire is the staged frame in its
// own format; the daemon does the conversion.
//
// Cost, and what follows from it. The network takes roughly 230 ms per frame on a 7900 XT, which is
// several frames of anybody's game, so the evaluate cannot be synchronous: the frame's proxy is copied
// to a readback buffer, the model runs on a worker thread, and the answer is copied back over the
// output some frames later. That is honest about what it is -- the composition is a few frames stale
// while the camera moves, and exact when it does not, which is what frame hold is for.

namespace DlssNr
{
namespace Hip
{

// The size the network runs at. Not a preference -- the weights are for this size.
constexpr unsigned int ModelWidth = 1920;
constexpr unsigned int ModelHeight = 1080;

// Whether this backend is switched on. Reads the config; does not load anything.
bool Enabled();

// Connects to the daemon. Safe to call every frame: it does the work once and then answers from what
// happened. False means this backend is unavailable and Status() says why -- most often that nothing
// is listening, because the game was launched without dlss5-nr-run.sh.
bool Ensure(ID3D12Device* device);

// Records the frame's half of the exchange on the caller's command list and returns the NGX-shaped
// result the pass expects: 1 on success, 0 when there is nothing to do or something failed.
//
// - modelInput is the proxy the model is shown, at exactly ModelWidth x ModelHeight, and in
//   NON_PIXEL_SHADER_RESOURCE -- the state the pass puts it in for the model, and the state this
//   restores it to after copying from it
// - output receives the model's answer, same size, same format
// - queue is the queue this list will be executed on; the exchange needs it to know when a copy it
//   recorded has actually run, and without it nothing can be handed to the model
//
// Success here means "the exchange is running", not "this frame's answer is in output": output
// carries the most recent finished answer, which is some frames old. On the first frames there is no
// answer yet and the call returns 0. The caller must read that as "leave this frame alone" and try
// again -- NOT as a failure. It used to read it as a failure, which disabled the pass on frame one of
// every session and made the whole backend look broken in every game but the one where the menu's
// Retry button could be clicked three times to walk it through priming. See Failed() below.
int Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* queue, ID3D12Resource* modelInput,
             ID3D12Resource* output, unsigned int workWidth, unsigned int workHeight, bool reset);

// Whether this backend has given up for the session.
//
// Evaluate returning 0 does not mean this. It means "output holds no answer this frame", which is the
// normal state of the first three frames: one to stage, one to submit, one to receive. Only this
// reports the exchange as actually broken, and only this should disable anything.
bool Failed();

// Drops everything: the worker, the staging buffers, the connection. Called on teardown and on a
// resolution or format change. The daemon keeps the weights loaded, so reconnecting is instant.
void Release();

// How many frames old the answer currently in the output texture is. Zero when there is none.
//
// The composition needs this to reproject: the model's answer is a picture of the world as it was this
// many frames ago, and laying it down unmoved is what smears.
unsigned int FramesSinceAnswer();

// Why the backend is unavailable, or what it is doing. Never null.
const char* Status();

// The wall-clock gap between the last two answers, smoothed. Zero until two have landed.
//
// This is not the same number as LastModelMs(), and the difference is the point. The model's round
// trip is how long one answer takes to make; the interval is how long each one is ON SCREEN, because
// the exchange is serial -- it stages, waits, delivers, waits, stages again -- and the delivered
// answer stays in the output texture until the next replaces it. A consumer that wants the age of
// what it is looking at needs modelMs + interval/2, not modelMs.
//
// SteadyStaleness() in the composition pass wanted exactly this and did not have it, so it used the
// model's cost as the answer's age and undercounted by 2.5x.
float LastAnswerIntervalMs();

// What the most recent Evaluate did on the game's list, for a caller that has to keep its own
// per-frame state in step with the exchange.
//
// StagedLastFrame() is the one that matters and the one that is easy to get wrong: the answer
// describes the scene as it was when the proxy was STAGED, not when it was delivered, so anything
// measuring how far the world has moved since must start counting there. Delivery is ~15 frames
// later, which is most of the displacement.
//
// Both are false on a frame that did neither, including every frame the backend is disabled,
// priming, or has failed on -- so a caller may read them unconditionally.
bool StagedLastFrame();
bool DeliveredLastFrame();

// Milliseconds the model took on its last completed frame, for the menu's timing table. Zero until
// one has completed.
float LastModelMs();

} // namespace Hip
} // namespace DlssNr
