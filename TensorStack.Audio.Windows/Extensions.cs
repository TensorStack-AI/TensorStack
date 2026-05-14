using System.Threading;
using System.Threading.Tasks;
using TensorStack.Audio.Windows;

namespace TensorStack.Audio
{
    public static class Extensions
    {
        /// <summary>
        /// Saves the Audio to file.
        /// </summary>
        /// <param name="audioStream">The audio stream.</param>
        /// <param name="audioFile">The audio file.</param>
        /// <param name="sampleRateOverride">The sample rate override.</param>
        /// <param name="channelsOverride">The channels override.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static Task SaveAync(this AudioInputStream audioStream, string audioFile, float? sampleRateOverride = default, int? channelsOverride = default, CancellationToken cancellationToken = default)
        {
            return AudioManager.SaveVideoStreamAync(audioFile, audioStream, sampleRateOverride, channelsOverride, cancellationToken);
        }
    }
}
