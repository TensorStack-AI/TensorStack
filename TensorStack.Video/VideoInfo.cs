// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Video
{
    public readonly record struct VideoInfo(string FileName, int Width, int Height, float FrameRate, int FrameCount)
    {
        public TimeSpan Duration => TimeSpan.FromSeconds(FrameCount / FrameRate);
    }
}
