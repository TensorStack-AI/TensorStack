// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Common.Vision
{
    public record CLIPImageOptions(int Width = 224, int Height = 224, ResizeMode ResizeMode = ResizeMode.Stretch, ResizeMethod ResizeMethod = ResizeMethod.Bilinear)
    {
        /// <summary>
        /// The Mean to use if normalizing the image.
        /// </summary>
        /// <value>The mean.</value>
        public float[] Mean { get; init; } = [0.485f, 0.456f, 0.406f];

        /// <summary>
        /// The Standard deviation to use if normalizing the image
        /// </summary>
        /// <value>The standard dev.</value>
        public float[] StdDev { get; init; } = [0.229f, 0.224f, 0.225f];
    }
}
