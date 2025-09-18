// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Tensor;

namespace TensorStack.Video
{
    public readonly record struct VideoInfo(string FileName, int Width, int Height, float FrameRate, int FrameCount, ImageTensor Thumbnail)
    {
        public TimeSpan Duration => TimeSpan.FromSeconds(FrameCount / FrameRate);
    }
}
