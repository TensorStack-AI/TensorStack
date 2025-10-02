// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Audio
{
    public record AudioInfo
    {
        public string FileName { get; init; }
        public int Channels { get; init; }
        public long Samples { get; init; }
        public int SampleRate { get; init; }
        public string AudioCodec { get; init; }
        public TimeSpan Duration { get; init; }
    }
}
