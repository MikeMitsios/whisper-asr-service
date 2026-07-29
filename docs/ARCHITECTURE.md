# Architecture

The design decisions here, and why each one is the way it is. The request-flow diagram lives in the
[README](../README.md#architecture); this document is the reasoning behind it.

---

## Layers

Three, with a deliberate dependency direction: the library knows nothing about HTTP.

```
app/                 HTTP concerns only: validation, caching, streaming, limits
  |
  v
src/whisper_asr/     Inference library. No FastAPI import anywhere.
  |
  v
transformers / optimum-quanto / torch
```

`whisper_asr` is importable and usable on its own — `scripts/` does exactly that. That separation is
what makes the API testable without a model and the library testable without a server.

| Component | Responsibility |
|---|---|
| `WhisperModel` | Loads weights, owns both inference backends, quantization, long-form windowing |
| `Transcriber` | Reads config, builds a `WhisperModel`, resamples, dispatches to a backend |
| `audio_utils` | Decoding from bytes, resampling with a fallback chain |
| `config` | YAML loading, defaults, strict validation |
| `evaluator` | Metrics over a sample set |
| `app.routers` | Request validation, cache protocol, SSE framing |
| `app.cache` | Fail-open async Redis |

---

## Why inference runs in a thread

`Transcriber.transcribe` is synchronous and CPU/GPU-bound. Called directly inside an `async def`
route it occupies the event loop for its entire duration, and nothing else in the process runs —
not other transcriptions, not `/health`, not the Redis calls that would have served a cache hit. A
single 30-second file makes the service look dead to a load balancer.

So the route does:

```python
async with request.app.state.inference_semaphore:
    text = await asyncio.to_thread(transcriber.transcribe, audio, sampling_rate)
```

`to_thread` moves the blocking work to a worker thread and the loop stays free. There is a test for
this: `test_inference_does_not_block_the_event_loop` fires a 1-second transcription and then asks
for `/health`, failing if the answer takes over 400 ms. In practice it comes back in about 1 ms.

The semaphore is the second half. There is one model instance, and two concurrent `generate` calls
on one model contend for the same weights and the same device — they do not run twice as fast, they
run slower than sequential and use twice the memory. `max_concurrent_inferences` defaults to `1`, so
requests queue rather than thrash. Making it explicit means the behaviour is a decision rather than
an accident.

The streaming route needs an async generator to do the same thing, since a sync generator cannot
`async with` a semaphore. Each `next()` on the token iterator is pushed through `asyncio.to_thread`.

---

## Why the cache key hashes decoded audio

The obvious cache key is a hash of the uploaded bytes. It is also nearly useless: the same recording
re-uploaded as MP3 instead of WAV, or at 44.1 kHz instead of 16 kHz, produces completely different
bytes and therefore a miss, even though the transcript would be identical.

So the key is computed from the audio, after decoding:

1. Decode the upload to a float32 waveform.
2. Resample to 16 kHz mono — the rate the model actually consumes.
3. Quantize to 16-bit PCM, a deterministic byte representation.
4. SHA-256 that.

Now a WAV and a lossless FLAC of one recording share a cache entry, and so do the same audio at
different container sample rates.

The limit is worth being precise about, because it is easy to overclaim: two files that *sound*
identical but hold different samples hash differently. A lossy re-encode does. So does handing the
same float input to two encoders that quantize it differently — WAV PCM_16 and FLAC do not agree on
rounding, which cost me a test that asserted otherwise. That is the correct outcome: those are
different samples, and the model may legitimately produce different text.

### The lock

On a miss, before running inference:

```
SET lock:asr:<hash> 1 NX EX 60
```

If two identical uploads arrive together, one wins the lock and transcribes; the other polls for the
result and returns it, or gives up with `503` after two seconds. Without this, N concurrent uploads
of the same file run N transcriptions and write the same answer N times.

The lock is released in a `finally`, so a failed transcription does not leave it held —
`test_lock_is_released_after_failure` covers that.

### Fail-open, and fail-fast

Every Redis operation is wrapped so that a failure degrades to "no cache" rather than to an error
response. Caching is an optimization; losing it should not lose the service.

Failing open only helps if it fails *quickly*, which is why the client is built with
`socket_connect_timeout` and `socket_timeout` of 2 seconds. Without them an unreachable Redis
stalls application startup and every subsequent lookup behind the client's default retry behaviour —
the fallback exists but you wait a long time to reach it.

---

## Long-form audio

Whisper's encoder consumes a fixed 3000-frame log-Mel spectrogram: exactly 30 seconds. The feature
extractor pads shorter audio up to that and, by default, **truncates anything longer**. A
six-minute upload silently returns the first thirty seconds with a `200 OK`.

There are two published ways to handle longer audio, and this project maps them onto its two
backends rather than picking a winner.

### `direct`: sequential

Feed window one. The model emits timestamp tokens interleaved with text. Read the last timestamp —
say `<|28.40|>` — and start window two there rather than at a blind 30 s, and pass window one's text
back as a decoder prompt so context survives the boundary. Repeat.

The result is that cuts land where the model thinks a segment ended, which is usually a pause, and
never mid-word. `transformers` implements this inside `generate()`; the work is in not defeating it:

```python
inputs = processor(audio, sampling_rate=sr, return_tensors="pt",
                   truncation=False, padding="longest", return_attention_mask=True)
generated = model.generate(
    input_features=..., attention_mask=...,
    return_timestamps=True, condition_on_prev_tokens=True,
    compression_ratio_threshold=1.35, logprob_threshold=-1.0,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)
```

`truncation=False` is the whole bug, in one keyword. The thresholds drive temperature fallback: a
window whose output looks degenerate — repetition loops show as a low compression ratio, low
confidence as a low mean logprob — is retried at a higher temperature. Temperature `0.0` is tried
first and almost always accepted, so the common case is unchanged.

Audio inside one window takes the original short path. One method, two branches, no duplicated
logic.

### `pipeline`: chunked and batched

The HuggingFace pipeline slices the waveform into 30-second windows with overlap on each side,
transcribes each independently, and stitches the overlaps by matching token sequences. Independent
windows batch, so this scales with `batch_size` and is the faster option on long files. It gives up
cross-window context, so accuracy is slightly lower.

This is a config passthrough — `chunk_length_s`, `stride_length_s`, `batch_size`.

### Streaming: a third path, and why

Streaming cannot reuse either of the above, for two reasons that compound.

**Windows must be timestamp-anchored, and reading a timestamp needs a finished decode.** Cutting
blindly every 30 seconds does not degrade gracefully. A window starting mid-utterance makes the
model predict end-of-sequence almost immediately: one window in testing returned **7 words for a
full 30 seconds of speech**, next to 121 for the window before it. Anchoring on the model's own
predicted timestamp fixes it, but the timestamp only exists once that window's generation is done.

**Temperature fallback runs several generate passes; the streamer closes after the first.**
`TextIteratorStreamer` receives tokens from attempt one and then `end()` fires, so the iterator
finishes while the retries keep running into a closed streamer. Measured directly: enabling fallback
with a streamer attached took 70 s instead of 15 s and returned byte-identical — degenerate — text.
It pays the retry cost and discards the retry.

So the split is by length:

- **Up to one window**: true token-by-token streaming, text identical to `transcribe_direct`.
- **Longer**: walk timestamp-anchored windows, emit each window's transcript as it completes, and
  strip the duplicated text at each seam.

A ten-minute upload produces output continuously instead of nothing until the end. Correct text at
window granularity beats corrupt text at token granularity, and the docstring on
`transcribe_direct_stream` says so, so nobody "fixes" it back.

### Seam de-duplication

Consecutive windows share audio, so their transcripts share text at the boundary. `strip_overlap`
finds the longest suffix of the accumulated text that is also a prefix of the new window's text and
removes it, comparing case- and punctuation-insensitively so `home.` still matches `Home`. It is a
pure function over two strings, so it is unit-tested directly rather than through the model.

---

## Configuration: strict, not forgiving

`load_config` raises on an unrecognised value for any constrained key — `device`,
`default_backend`, `long_form_mode` — and `resolve_dtype` raises on an unknown dtype. There is no
fallback to a default.

This is a reaction to a specific failure. `_DTYPE_MAP.get(name, torch.float32)` meant a config
asking for `torch_dtype: float8` got float32 without a word, and two rows of the results table were
recorded under a precision that never ran. A silent fallback does not prevent a bug; it converts a
startup error into wrong data you find out about much later. The
[experiments write-up](EXPERIMENTS.md#notes-on-data-integrity) has the details.

Deprecated keys are the one exception. `enable_callibration` — the original misspelling — is
migrated to `enable_calibration` with a `DeprecationWarning`, because it was a documented public
name and silently ignoring it would be worse than either alternative.

---

## Startup and imports

The model is built in the FastAPI lifespan, not at import. `Transcriber` is imported inside the
lifespan function too, so `import app.main` pulls in neither `transformers` nor `optimum-quanto` nor
`datasets`.

That is not tidiness, it is what makes the test suite viable. Importing `app.main` used to download
and load model weights, so no test could import the app without a GPU, a network connection and a
few minutes. It is now 2.5 s and side-effect free, and CI asserts both properties: that
`app.state.transcriber` does not exist after import, and that those three modules are absent from
`sys.modules`.

`whisper_asr/__init__.py` uses [PEP 562](https://peps.python.org/pep-0562/) module-level
`__getattr__` for the same reason. `from whisper_asr.config import load_config` costs 0.02 s and
imports no torch, while `from whisper_asr import Transcriber` still works and pulls in what it
needs on first access.

---

## Testing strategy

Three tiers, separated by pytest markers, with the default selection excluding the slow two.

**Default — offline, no weights, no server.** Audio is synthesized with numpy and soundfile; a
`FakeTranscriber` is injected onto `app.state`; requests go through `TestClient` in-process. 83
tests in about 8 seconds. This tier has to stay free of network and weights, or CI stops being
useful.

Notably, `test_streaming_isolation.py` drives the real streaming code with a stub model and
tokenizer. That is what pins the shared-streamer bug: a single `TextIteratorStreamer` built in
`__init__` and reused per call meant two concurrent streams consumed each other's tokens. The test
runs several streams at once, each emitting a distinguishable token, and fails if any result
contains more than one.

**`-m slow` — real weights.** `openai/whisper-tiny` on CPU, roughly two minutes. This is where
long-form behaviour is verified end to end. The key assertion is on **mel frame count**, 3000
against 7500, not on transcript length — the fixture is synthesized rather than real speech, so the
model hallucinates on it, and the truncating path happily produces a *longer* repetition loop than
the fixed path. Frame count is where the bug actually lived, so frame count is what the test checks.

**`-m integration` — real Redis.** Cache hits, container-independent keys, no collisions, failures
not cached, locks released. Skips cleanly when Redis is absent.

No Emilia-Dataset audio is committed. It is CC BY-NC with additional terms, so redistributing it in
a public repository is not on, and synthesized audio makes the suite hermetic anyway.

---

## What is deliberately not here

- **Cross-request batching.** The throughput win, and the largest missing piece. Needs a queue that
  groups arrivals into a window and runs them as one batch.
- **VAD pre-segmentation.** Silero VAD ahead of the windowing would skip silence entirely — a large
  win on sparse audio, at the cost of a second model.
- **A metrics endpoint.** Per-response timings exist; aggregation does not.
- **Multiple models per process.** One `Transcriber` on `app.state`. Serving several would mean
  reworking the semaphore into a per-model pool.
