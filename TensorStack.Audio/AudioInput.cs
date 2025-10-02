// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;

namespace TensorStack.Audio
{
    public abstract class AudioInputBase : AudioTensor
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInputBase"/> class.
        /// </summary>
        /// <param name="audioTensor">The audio tensor.</param>
        protected AudioInputBase(AudioTensor audioTensor)
            : base(audioTensor, audioTensor.SampleRate) { }

        /// <summary>
        /// Gets the source audio filename.
        /// </summary>
        public abstract string SourceFile { get; }

        /// <summary>
        /// Save the Audio to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        public abstract void Save(string filename);

        /// <summary>
        /// Save the Audio to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token</param>
        public abstract Task SaveAsync(string filename, CancellationToken cancellationToken = default);
    }
}
