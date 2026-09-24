using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using ManagedBass;
using osu.Framework.Audio.Callbacks;
using ParaTactus;

namespace ParaTactus.Decoding
{
    /// <summary>
    /// Decoding audio to the mono sample rate the beat model works at, with BASS.
    /// </summary>
    /// <remarks>
    /// This is the optional half of the library, and it is separate for one reason: BASS is not open source. It is free
    /// for non-commercial use and licensed per product otherwise, and its terms say a licensed product must be an
    /// end-user product rather than a component used by other products. Keeping it out of <c>ParaTactus</c> means the
    /// analysis can be used, published and built on with no proprietary dependency at all; keeping it here means
    /// callers who want a path decoded for them still have one call to make, with the terms visible in the package
    /// they chose to reference instead of hidden in a transitive one.
    ///
    /// The BASS error codes are part of the public surface of <see cref="DecodeException"/>, so this assembly - unlike
    /// the one it extends - carries a dependency whose terms are not the MIT ones of its own source. See
    /// <c>THIRD-PARTY-NOTICES.md</c> beside it.
    ///
    /// The namespace here is <c>ParaTactus.Decoding</c> while the package is <c>ParaTactus.Bass</c>, and that is not
    /// arbitrary. A namespace ending in the same word as a type shadows it for everything declared inside it: with
    /// <c>ParaTactus.Bass</c> present, <c>Bass.CreateStream</c> in this file - and <c>Bass.Init</c> in anything
    /// declared under a <c>ParaTactus.*</c> namespace, which includes the whole test suite - resolves to the namespace
    /// rather than to <c>ManagedBass.Bass</c>, and does not compile. A using directive does not help, because
    /// members of an enclosing namespace take precedence over using directives.
    /// </remarks>
    public sealed class BassAudioDecoder : IAudioDecoder
    {

        private static readonly object startLock = new object();
        private static bool started;

        /// <summary>
        /// Starts BASS if nothing else has, once per process.
        /// </summary>
        /// <remarks>
        /// Decoding needs BASS started, and a library that returns decoded samples should not require its caller to
        /// know that. Nothing here did, and the gap was invisible until something stood on its own: the test suite has
        /// its own environment helper and an application on osu.Framework gets it from the framework, so only a
        /// standalone consumer saw <c>Errors.Init</c> - which reads like a missing codec rather than an unstarted
        /// audio system.
        ///
        /// The no-sound device is deliberate: decoding does not need an output device, and asking for one would fail
        /// on a machine without audio hardware or in a headless process.
        /// </remarks>
        private static void ensureStarted()
        {
            if (started)
                return;

            lock (startLock)
            {
                if (started)
                    return;

                if (!Bass.Init(Bass.NoSoundDevice, 44100, DeviceInitFlags.Default, IntPtr.Zero)
                    && Bass.LastError != Errors.Already)
                {
                    throw new DecodeException("<init>", Bass.LastError);
                }

                started = true;
            }
        }
        /// <summary>
        /// The rate everything downstream works at, mirroring <see cref="AnalysisAudio.SampleRate"/> in the analysis so
        /// that call sites read as one decision rather than two.
        /// </summary>
        public const int ANALYSIS_SAMPLE_RATE = AnalysisAudio.SampleRate;

        /// <summary>
        /// An instance for the places that want a decoder rather than samples.
        /// </summary>
        /// <remarks>
        /// Typed as the interface on purpose, so that <c>Default.DecodeMono</c> is unambiguous: the same signature
        /// exists as a static on this class, and a member access on a class-typed value would find neither.
        /// </remarks>
        public static readonly IAudioDecoder Default = new BassAudioDecoder();

        /// <inheritdoc />
        float[] IAudioDecoder.DecodeMono(string path, int sampleRate, CancellationToken cancellation)
            => DecodeMono(path, sampleRate, cancellation);

        /// <summary>
        /// Decodes any BASS-supported stream at its native rate, reporting the channel layout.
        /// </summary>
        /// <remarks>
        /// Resampling and channel mixing are deliberately left to <see cref="MonoMixdown"/>. Asking BASS to resample a
        /// decode stream is not dependable - in testing it returned twice the expected number of frames, i.e. it
        /// decoded at the source rate while the caller believed otherwise, which silently doubles every tempo.
        /// Native-rate decode plus an explicit mixdown and decimation has no such ambiguity.
        ///
        /// Takes a path rather than a stream wherever one is available. Decoding through file callbacks with
        /// <see cref="StreamSystem.NoBuffer"/> works for simple containers but asks the decoder to cope without a
        /// seekable input, which compressed formats with a header to skip - an MP3 with an ID3v2 tag, a VBR header, or
        /// a trailing tag - can reasonably refuse. BASS's own file reader handles all of that.
        /// </remarks>
        public static RawPcm DecodeRaw(string path, CancellationToken cancellation = default)
        {
            ensureStarted();

            int handle = Bass.CreateStream(path, 0, 0, BassFlags.Decode | BassFlags.Float);

            try
            {
                if (handle == 0)
                    throw new DecodeException(path, Bass.LastError);

                return readAll(handle, cancellation);
            }
            finally
            {
                if (handle != 0)
                    Bass.StreamFree(handle);
            }
        }

        /// <summary>
        /// Decodes a stream that has no path - a beatmap resource, or a caller holding a stream. Falls back to
        /// file callbacks, which is the only option available in that case.
        /// </summary>
        public static RawPcm DecodeRaw(Stream stream, CancellationToken cancellation = default)
        {
            ensureStarted();

            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            using (var callbacks = new FileCallbacks(new DataStreamFileProcedures(stream)))
            {
                int handle = Bass.CreateStream(StreamSystem.NoBuffer, BassFlags.Decode | BassFlags.Float, callbacks.Callbacks, callbacks.Handle);

                try
                {
                    if (handle == 0)
                        throw new DecodeException("<stream>", Bass.LastError);

                    return readAll(handle, cancellation);
                }
                finally
                {
                    if (handle != 0)
                        Bass.StreamFree(handle);
                }
            }
        }

        /// <summary>
        /// Decodes an audio file to mono float PCM at the requested rate, using BASS's own file reader.
        /// </summary>
        public static float[] DecodeMono(string path, int sampleRate = ANALYSIS_SAMPLE_RATE, CancellationToken cancellation = default)
            => MonoMixdown.ToMono(DecodeRaw(path, cancellation), sampleRate);

        /// <summary>
        /// Decodes any BASS-supported stream to mono float PCM at the requested rate.
        /// </summary>
        public static float[] DecodeMono(Stream stream, int sampleRate = ANALYSIS_SAMPLE_RATE, CancellationToken cancellation = default)
            => MonoMixdown.ToMono(DecodeRaw(stream, cancellation), sampleRate);

        /// <summary>
        /// Pulls the whole decoded channel into memory.
        /// </summary>
        private static RawPcm readAll(int handle, CancellationToken cancellation)
        {
            var info = Bass.ChannelGetInfo(handle);
            int channels = Math.Max(1, info.Channels);
            int sampleRate = info.Frequency > 0 ? info.Frequency : ANALYSIS_SAMPLE_RATE;

            var chunk = new float[1 << 16];
            var output = new List<float>(1 << 22);

            while (true)
            {
                cancellation.ThrowIfCancellationRequested();

                int read = Bass.ChannelGetData(handle, chunk, chunk.Length * sizeof(float));

                if (read <= 0)
                    break;

                int sampleCount = read / sizeof(float);
                output.AddRange(chunk.AsSpan(0, sampleCount).ToArray());
            }

            return new RawPcm(output.ToArray(), channels, sampleRate);
        }

        /// <summary>
        /// A decode failure, carrying the BASS error so the caller can report something actionable instead of
        /// collapsing every cause into "could not decode".
        /// </summary>
        public class DecodeException : Exception
        {
            public DecodeException(string source, Errors error)
                : base($"BASS could not decode {source}: {error} ({(int)error}). "
                       + "Errors.Init means BASS was never initialised, which normally means the audio system "
                       + "failed to start; Errors.FileFormat means the codec is not supported; Errors.FileOpen "
                       + "means the file could not be read.")
            {
                Error = error;
            }

            /// <summary>The underlying BASS error code.</summary>
            public Errors Error { get; }
        }
    }
}
