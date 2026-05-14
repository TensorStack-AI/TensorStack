// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Audio
{
    public class AudioInputStreamBase
    {
        private readonly AudioInfo _audioInfo;

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInputStreamBase"/> class.
        /// </summary>
        /// <param name="audioInfo">The audio information.</param>
        public AudioInputStreamBase(AudioInfo audioInfo)
        {
            _audioInfo = audioInfo;
        }

        /// <summary>
        /// Gets the filename.
        /// </summary>
        /// <value>The filename.</value>
        public string SourceFile => _audioInfo.FileName;

        /// <summary>
        /// Gets the audio sample rate.
        /// </summary>
        public int SampleRate => _audioInfo.SampleRate;

        /// <summary>
        /// Gets the count of samples.
        /// </summary>
        public long Samples => _audioInfo.Samples;

        /// <summary>
        /// Gets the audio channel count.
        /// </summary>
        public int Channels => _audioInfo.Channels;

        /// <summary>
        /// Gets the audio codec.
        /// </summary>
        public string AudioCodec => _audioInfo.AudioCodec;

        /// <summary>
        /// Gets the duration.
        /// </summary>
        public TimeSpan Duration => _audioInfo.Duration;

    }
}
