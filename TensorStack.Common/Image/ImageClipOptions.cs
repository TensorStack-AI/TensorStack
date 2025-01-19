// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Common.Image
{
    public record ImageClipOptions
    {
        public int Width { get; init; } = 224;
        public int Height { get; init; } = 224;
        public float[] Mean { get; init; } = [0.485f, 0.456f, 0.406f];
        public float[] StdDev { get; init; } = [0.229f, 0.224f, 0.225f];
    }
}
