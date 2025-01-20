// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace TensorStack.Upscaler.Config
{
    /// <summary>
    /// Default UpscalerConfig.
    /// </summary>
    public record UpscalerConfig(string Path, Provider Provider = Provider.CPU, int DeviceId = 0, bool IsOptimizationSupported = true)
        : ModelConfig(Path, Provider, DeviceId, IsOptimizationSupported)
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
        /// The scale factor the model supports, 2x 4x etc
        /// </summary>
        public int ScaleFactor { get; init; } = 1;

        /// <summary>
        /// The models expected input normalization (0-1 or -1-1)
        /// </summary>
        public Normalization Normalization { get; init; }
    }
}
