// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace TensorStack.Extractors.Common
{
    public record ExtractorConfig : ModelConfig
    {
        /// <summary>
        /// The channels the model supports 1 = Greyscale, RGB = 3, RGBA = 4.
        /// </summary>
        public int Channels { get; init; } = 3;

        /// <summary>
        /// The models input maximum size (0 = Any)
        /// </summary>
        public int SampleSize { get; init; }

        /// <summary>
        /// The models expected input normalization (0-1 or -1-1)
        /// </summary>
        public Normalization Normalization { get; init; } = Normalization.ZeroToOne;

        /// <summary>
        /// The models required output normalization
        /// </summary>
        public Normalization OutputNormalization { get; init; }

        /// <summary>
        /// If the result should ne inverted
        /// </summary>
        public bool IsOutputInverted { get; init; }

        /// <summary>
        /// The channels the model supports 1 = Greyscale, RGB = 3, RGBA = 4.
        /// </summary>
        public int OutputChannels { get; init; } = 1;
    }
}
