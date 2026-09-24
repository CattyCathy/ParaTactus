using System.Threading;

namespace ParaTactus
{
    /// <summary>
    /// Decoded PCM exactly as the source stores it, without resampling or channel mixing.
    /// </summary>
    public readonly record struct RawPcm(float[] Samples, int Channels, int SampleRate);

    /// <summary>
    /// Turning an audio file into the mono samples the model works at.
    /// </summary>
    /// <remarks>
    /// The analysis is defined on samples, not on files, and this interface is where that boundary is drawn. Everything
    /// downstream - the frontend, the model, the peak picking, the regulariser, the grid - is arithmetic on a
    /// <c>float[]</c>, and none of it cares what produced one.
    ///
    /// The interface exists because of licensing rather than architecture. The decoder this project was developed
    /// against is BASS, reached through <c>osu.Framework</c>, and BASS is proprietary: free for non-commercial use and
    /// licensed per product otherwise. A library that cannot be used without it cannot honestly be called MIT, so the
    /// decode moved behind this interface and out of this assembly, into <c>ParaTactus.Bass</c>. See
    /// <c>THIRD-PARTY-NOTICES.md</c>.
    ///
    /// Implementations resample and mix down themselves, or use <see cref="MonoMixdown"/> to do it. A caller that
    /// already has samples never needs an implementation at all.
    /// </remarks>
    public interface IAudioDecoder
    {
        /// <summary>
        /// The file decoded to mono at the requested rate.
        /// </summary>
        float[] DecodeMono(string path, int sampleRate, CancellationToken cancellation = default);
    }
}
