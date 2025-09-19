// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Tensor;

namespace TensorStack.Video
{
    public record VideoInfo
    {
        public string FileName { get; init; }
        public int Width { get; init; }
        public int Height { get; init; }
        public float FrameRate { get; init; }
        public int FrameCount { get; init; }
        public string VideoCodec { get; init; }
        public ImageTensor Thumbnail { get; init; }
        public TimeSpan Duration => TimeSpan.FromSeconds(FrameCount / FrameRate);
    }
}
