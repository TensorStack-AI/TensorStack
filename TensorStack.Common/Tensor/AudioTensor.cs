// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Common.Tensor
{
    /// <summary>
    /// Class to handle audio in Tensor format
    /// Implements the <see cref="Tensor{float}" />
    /// </summary>
    /// <seealso cref="Tensor{float}" />
    public class AudioTensor : Tensor<float>
    {
        protected int _sampleRate;
        protected TimeSpan _duration;

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioTensor"/> class.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="sampleRate">The source audio sample rate.</param>
        public AudioTensor(Tensor<float> tensor, int sampleRate = 16000)
            : base(tensor.Memory, tensor.Dimensions)
        {
            _sampleRate = sampleRate;
            ThrowIfInvalid();
        }

        /// <summary>
        /// Gets the audio channel count (Mono, Stereo etc)
        /// </summary>
        public int Channels => Dimensions[0];

        /// <summary>
        /// Gets the sample count.
        /// </summary>
        public int Samples => Dimensions[1];

        /// <summary>
        /// Gets the sample rate.
        /// </summary>
        public int SampleRate => _sampleRate;

        /// <summary>
        /// Gets the duration.
        /// </summary>
        public TimeSpan Duration => TimeSpan.FromSeconds((double)Samples / SampleRate);


        /// <summary>
        /// Normalizes the tensor values from range -1 to 1 to 0 to 1.
        /// </summary>
        public void NormalizeZeroToOne()
        {
            this.NormalizeZeroOne();
        }


        /// <summary>
        /// Normalizes the tensor values from range 0 to 1 to -1 to 1.
        /// </summary>
        public void NormalizeOneToOne()
        {
            this.NormalizeOneOne();
        }


        /// <summary>
        /// Throws if Dimensions are invalid.
        /// </summary>
        protected void ThrowIfInvalid()
        {
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Samples, 0, nameof(Samples));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Channels, 0, nameof(Channels));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(SampleRate, 0, nameof(SampleRate));
        }
    }
}
