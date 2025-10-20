// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Extractors.Common
{
    public record PoseOptions : IRunOptions
    {
        /// <summary>
        /// Max body detections (0 = Auto).
        /// </summary>
        public int Detections { get; set; } = 0;

        /// <summary>
        /// Gets the body confidence.
        /// </summary>
        public float BodyConfidence { get; init; } = 0.4f;

        /// <summary>
        /// Gets the joint confidence.
        /// </summary>

        public float JointConfidence { get; init; } = 0.1f;

        /// <summary>
        /// Gets the skeleton color alpha.
        /// </summary>
        public float ColorAlpha { get; init; } = 0.8f;

        /// <summary>
        /// Gets the joint elipse radius.
        /// </summary>
        public float JointRadius { get; init; } = 7f;

        /// <summary>
        /// Gets the bone radius.
        /// </summary>
        public float BoneRadius { get; init; } = 8f;

        /// <summary>
        /// Gets the bone thickness.
        /// </summary>
        public float BoneThickness { get; init; } = 1f;

        /// <summary>
        /// Gets or sets if the background is Black or Transparent
        /// </summary>
        public bool IsTransparent{ get; set; }
    }


    /// <summary>
    /// Image PoseOptions.
    /// </summary>
    public record PoseImageOptions : PoseOptions
    {
        /// <summary>
        /// Gets the input.
        /// </summary>
        public ImageTensor Image { get; init; }
    }


    /// <summary>
    /// Video PoseOptions.
    /// </summary>
    public record PoseVideoOptions : PoseOptions
    {
        /// <summary>
        /// Gets the input.
        /// </summary>
        public VideoTensor Video { get; init; }
    }


    /// <summary>
    /// Stream PoseOptions.
    /// </summary>
    public record PoseStreamOptions : PoseOptions
    {
        /// <summary>
        /// Gets the input.
        /// </summary>
        public IAsyncEnumerable<VideoFrame> Stream { get; init; }
    }
}
