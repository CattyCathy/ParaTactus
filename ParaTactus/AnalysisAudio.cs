namespace ParaTactus
{
    /// <summary>
    /// The sample rate everything here works at.
    /// </summary>
    /// <remarks>
    /// Fixed rather than configurable, because the model's frame grid is defined in terms of it: the frontend hops 441
    /// samples at 22050Hz, which is exactly 50 frames per second, and that is the rate the model's output is in. Any
    /// other rate would need the whole frontend re-derived.
    /// </remarks>
    public static class AnalysisAudio
    {
        /// <summary>The rate the analysis runs at, in hertz.</summary>
        public const int SampleRate = 22050;
    }
}
