// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using OpenCvSharp;
using System;
using TensorStack.Common.Tensor;

namespace TensorStack.Video
{
    public static class Extensions
    {
        /// <summary>
        /// Converts Matrix to Tensor.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static unsafe ImageTensor ToTensor(this Mat matrix, Size cropSize = default)
        {
            int cropX = 0;
            int cropY = 0;
            int height = matrix.Rows;
            int width = matrix.Cols;

            if (cropSize != default)
            {
                if (width == cropSize.Width)
                {
                    cropY = (height - cropSize.Height) / 2;
                    height = cropSize.Height;
                }
                else if (height == cropSize.Height)
                {
                    cropX = (width - cropSize.Width) / 2;
                    width = cropSize.Width;
                }
            }

            var imageTensor = new ImageTensor([1, 4, height, width]);
            var destination = imageTensor.Memory.Span;

            unsafe
            {
                var source = matrix.DataPointer;
                int srcStride = matrix.Cols * 3;
                int dstStride = height * width;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int srcIndex = ((y + cropY) * matrix.Cols + (x + cropX)) * 3;
                        int dstIndex = y * width + x;

                        destination[0 * dstStride + dstIndex] = GetFloatValue(source[srcIndex + 2]); // R
                        destination[1 * dstStride + dstIndex] = GetFloatValue(source[srcIndex + 1]); // G
                        destination[2 * dstStride + dstIndex] = GetFloatValue(source[srcIndex + 0]); // B
                        destination[3 * dstStride + dstIndex] = GetFloatValue(byte.MaxValue);        // A
                    }
                }
            }

            return imageTensor;
        }


        /// <summary>
        /// Converts Tensor to OpenCv Matrix.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Mat.</returns>
        internal static unsafe Mat ToMatrix(this Tensor<float> tensor)
        {
            var channels = tensor.Dimensions[1];
            var height = tensor.Dimensions[2];
            var width = tensor.Dimensions[3];

            var matrix = new Mat(height, width, MatType.CV_8UC3);
            var source = tensor.Span;
            var destination = matrix.DataPointer;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int offset = y * width + x;

                    if (channels == 1)
                    {
                        byte gray = GetByteValue(source[offset]);
                        destination[offset * 3 + 0] = gray; // B
                        destination[offset * 3 + 1] = gray; // G
                        destination[offset * 3 + 2] = gray; // R
                    }
                    else
                    {
                        destination[offset * 3 + 0] = GetByteValue(source[2 * width * height + offset]); // B
                        destination[offset * 3 + 1] = GetByteValue(source[1 * width * height + offset]); // G
                        destination[offset * 3 + 2] = GetByteValue(source[0 * width * height + offset]); // R
                    }
                }
            }

            return matrix;
        }


        /// <summary>
        /// Gets the normalized byte value.
        /// </summary>
        /// <param name="value">The value.</param>
        internal static byte GetByteValue(this float value)
        {
            return (byte)Math.Clamp((value + 1.0f) * 0.5f * 255f, 0, 255);
        }


        /// <summary>
        /// Gets the normalized float value.
        /// </summary>
        /// <param name="value">The value.</param>
        internal static float GetFloatValue(this byte value)
        {
            return (value / 255f) * 2.0f - 1.0f;
        }


        /// <summary>
        /// Null if zero.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>System.Nullable&lt;System.Int32&gt;.</returns>
        internal static int? NullIfZero(this int? value)
        {
            if (value.HasValue && value.Value == 0)
                return null;

            return value;
        }

    }
}
