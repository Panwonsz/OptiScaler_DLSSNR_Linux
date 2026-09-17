#pragma once

#include <d3d12.h>

// DLSS 5 Neural Rendering on AMD, through a native HIP implementation of the model.
//
// This is a second backend for the one call in this module that is NVIDIA-only. Everything else the
// pass does -- building the display-referred proxy, the exposure work, the HUD mask, the resolve that
// composes the answer back, capture, frame hold, the menu -- is API-agnostic and runs unchanged. Only
// the model itself cannot run through NGX on a Radeon, and that is the piece replaced here.
//
// The model is the same network, re-implemented in HIP for RDNA 3/4 and driven from the Linux side of
// a Proton process:
//
//   libdlss5_hip.so   the network, loaded into the Wine process by LD_PRELOAD, running on the real
//                     GPU through ROCm; it publishes a table of function pointers in the Unix
//                     environment when it loads
//   dlss5_hip.dll     a small PE trampoline that reads that table through Wine's live Unix
//                     environment and calls into it; this is what the code here loads
//
// The contract is deliberately narrow: packed float RGBA in, packed float RGB out, 1920x1080, which is
// the size the network was built for. The caller therefore has to drive the pass at exactly that
// working size -- see DlssNrHipModelWidth/Height.
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

// Loads dlss5_hip.dll and initialises the model. Safe to call every frame: it does the work once and
// then answers from what happened. False means this backend is unavailable and Status() says why.
bool Ensure(ID3D12Device* device);

// Records the frame's half of the exchange on the caller's command list and returns the NGX-shaped
// result the pass expects: 1 on success, 0 when there is nothing to do or something failed.
//
// - modelInput is the proxy the model is shown, at exactly ModelWidth x ModelHeight
// - output receives the model's answer, same size, same format
// - queue is the queue this list will be executed on; the exchange needs it to know when a copy it
//   recorded has actually run, and without it nothing can be handed to the model
//
// Success here means "the exchange is running", not "this frame's answer is in output": output
// carries the most recent finished answer, which is some frames old. On the first frames there is no
// answer yet and the call returns 0, which the pass already treats as "leave the frame alone".
int Evaluate(ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* queue, ID3D12Resource* modelInput,
             ID3D12Resource* output, unsigned int workWidth, unsigned int workHeight, bool reset);

// Drops everything: the worker, the staging buffers, the model. Called on teardown and on a
// resolution or format change.
void Release();

// Why the backend is unavailable, or what it is doing. Never null.
const char* Status();

// Milliseconds the model took on its last completed frame, for the menu's timing table. Zero until
// one has completed.
float LastModelMs();

} // namespace Hip
} // namespace DlssNr
