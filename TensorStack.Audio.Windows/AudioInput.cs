// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Audio.Windows;
using TensorStack.Common.Tensor;

namespace TensorStack.Audio
{
    /// <summary>
    /// Class to handle processing of a audio from file.
    /// </summary>
    public class AudioInput : AudioInputBase
    {
        private string _sourceFile;

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public AudioInput(string filename, string audioCodec = "pcm_s16le", int sampleRate = 16000, int channels = 1)
            : this(filename, AudioManager.LoadTensor(filename, audioCodec, sampleRate, channels)) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInput"/> class.
        /// </summary>
        /// <param name="audioTensor">The audio tensor.</param>
        public AudioInput(AudioTensor audioTensor)
            : base(audioTensor) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="audioTensor">The audio tensor.</param>
        protected AudioInput(string filename, AudioTensor audioTensor)
            : base(audioTensor)
        {
            _sourceFile = filename;
        }

        /// <summary>
        /// Gets the source audio filename.
        /// </summary>
        public override string SourceFile => _sourceFile;


        /// <summary>
        /// Save the Audio to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        public override void Save(string filename)
        {
            if (string.IsNullOrEmpty(_sourceFile))
                _sourceFile = filename;

            AudioManager.SaveAudio(filename, this);
        }


        /// <summary>
        /// Save the Audio to file asynchronously
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public override async Task SaveAsync(string filename, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(_sourceFile))
                _sourceFile = filename;

            await AudioManager.SaveAudioAync(filename, this, cancellationToken);
        }


        /// <summary>
        /// Create a AudioInput asynchronously
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task<AudioInput> CreateAsync(string filename, string audioCodec = "pcm_s16le", int sampleRate = 16000, int channels = 1, CancellationToken cancellationToken = default)
        {
            return new AudioInput(filename, await AudioManager.LoadTensorAsync(filename, audioCodec, sampleRate, channels, cancellationToken));
        }
    }
}
