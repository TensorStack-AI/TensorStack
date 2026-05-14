// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Audio.Windows;

namespace TensorStack.Audio
{
    public class AudioInputStream : AudioInputStreamBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInputStream"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <exception cref="System.Exception">Failed to open audio file.</exception>
        public AudioInputStream(string filename)
            : this(AudioManager.LoadInfo(filename)) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInputStream"/> class.
        /// </summary>
        /// <param name="audioInfo">The audio information.</param>
        private AudioInputStream(AudioInfo audioInfo)
            : base(audioInfo) { }


        /// <summary>
        /// Get an AudioInput from the stream
        /// </summary>
        /// <param name="sampleRate">The sample rate.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="audioCodec">The audio codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public async Task<AudioInput> GetAsync(int sampleRate = 48000, int channels = 1, string audioCodec = "pcm_s16le", CancellationToken cancellationToken = default)
        {
            return new AudioInput(await AudioManager.LoadTensorAsync(SourceFile, audioCodec, sampleRate, channels, cancellationToken));
        }


        /// <summary>
        /// Move a AudioInputStream asynchronously
        /// </summary>
        /// <param name="newFilename">The new filename.</param>
        /// <param name="overwrite">if set to <c>true</c> overwrite destination.</param>
        public async Task<AudioInputStream> MoveAsync(string newFilename, bool overwrite = true)
        {
            if (!File.Exists(SourceFile))
                throw new Exception("Source audio not found");

            File.Move(SourceFile, newFilename, overwrite);
            return await CreateAsync(newFilename);
        }


        /// <summary>
        /// Copy a AudioInputStream asynchronously
        /// </summary>
        /// <param name="newFilename">The new filename.</param>
        /// <param name="overwrite">if set to <c>true</c> overwrite destination.</param>
        public async Task<AudioInputStream> CopyAsync(string newFilename, bool overwrite = true)
        {
            if (!File.Exists(SourceFile))
                throw new Exception("Source audio not found");

            File.Copy(SourceFile, newFilename, overwrite);
            return await CreateAsync(newFilename);
        }


        /// <summary>
        /// Create a AudioInputStream asynchronously
        /// </summary>
        /// <param name="filename">The filename.</param>
        public static async Task<AudioInputStream> CreateAsync(string filename)
        {
            return new AudioInputStream(await AudioManager.LoadInfoAsync(filename));
        }
    }
}
