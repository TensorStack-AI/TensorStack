// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using TensorStack.Common.Tensor;

namespace TensorStack.Common.Image
{
    public static class ImageTiles
    {
        /// <summary>
        /// Computes the tiles.
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="tileSize">Size of the tile.</param>
        /// <param name="maxTileSize">Maximum size of the tile.</param>
        /// <returns>List&lt;TileJob&gt;.</returns>
        public static List<TileJob> ComputeTiles(ImageTensor inputImage, int tileSize, int maxTileSize)
        {
            var ys = ComputeOffsets(inputImage.Height, tileSize);
            var xs = ComputeOffsets(inputImage.Width, tileSize);
            var imageTiles = new List<TileJob>();
            foreach (int y in ys)
                foreach (int x in xs)
                    imageTiles.Add(new TileJob(x, y, maxTileSize));

            return imageTiles;
        }


        /// <summary>
        /// Compute tile offsets (step = tile - overlap). Last tile clamped backward.
        /// </summary>
        /// <param name="full">The full.</param>
        /// <param name="step">The step.</param>
        private static int[] ComputeOffsets(int full, int step)
        {
            if (full <= step)
                return [0];

            var list = new List<int>();
            for (int pos = 0; pos + step < full; pos += step)
                list.Add(pos);

            int last = full - step;
            if (list[^1] != last)
                list.Add(last);

            return list.ToArray();
        }


        /// <summary>
        /// Creates the weight sum.
        /// </summary>
        /// <param name="outputImage">The output image.</param>
        public static Tensor<float> CreateWeightSum(ImageTensor outputImage)
        {
            return new Tensor<float>([1, 1, outputImage.Height, outputImage.Width]);
        }


        /// <summary>
        /// Weighted feathering map for tile blending
        /// </summary>
        /// <param name="tileSize">Size of the tile.</param>
        /// <param name="overlap">The overlap.</param>
        public static float[,] CreateWeightMap(int tileSize, int overlap)
        {
            var w = new float[tileSize, tileSize];
            for (int y = 0; y < tileSize; y++)
            {
                float wy = 1f;
                if (y < overlap)
                    wy = (float)y / overlap;
                else if (y >= tileSize - overlap)
                    wy = (float)(tileSize - y - 1) / overlap;

                for (int x = 0; x < tileSize; x++)
                {
                    float wx = 1f;
                    if (x < overlap)
                        wx = (float)x / overlap;
                    else if (x >= tileSize - overlap)
                        wx = (float)(tileSize - x - 1) / overlap;

                    w[y, x] = MathF.Min(wx, wy);
                }
            }
            return w;
        }


        /// <summary>
        /// Extracts the tile span.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="posX">The position x.</param>
        /// <param name="posY">The position y.</param>
        /// <param name="tileSize">Size of the tile.</param>
        /// <param name="channels">The channels.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> ExtractTileSpan(Tensor<float> imageTensor, int posX, int posY, int tileSize, int channels)
        {
            int height = imageTensor.Dimensions[2];
            int width = imageTensor.Dimensions[3];
            var tileShape = new[] { 1, channels, tileSize, tileSize };
            var tileSpan = new TensorSpan<float>(tileShape);
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < tileSize; y++)
                {
                    int srcY = Math.Min(posY + y, height - 1);
                    for (int x = 0; x < tileSize; x++)
                    {
                        int srcX = Math.Min(posX + x, width - 1);
                        tileSpan[0, c, y, x] = imageTensor[0, c, srcY, srcX];
                    }
                }
            }
            return tileSpan;
        }


        /// <summary>
        /// Blends the tile into the outputImage.
        /// </summary>
        /// <param name="outputImage">The output image.</param>
        /// <param name="weightSum">The weight sum.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="weight">The weight.</param>
        /// <param name="posX">The position x.</param>
        /// <param name="posY">The position y.</param>
        public static void BlendTile(ImageTensor output, Tensor<float> weightSum, Tensor<float> tile, float[,] weight, int posX, int posY, int scaleFactor)
        {
            var channels = tile.Dimensions[1];
            var tileSize = tile.Dimensions[2];

            var outH = output.Height;
            var outW = output.Width;

            var outHW = outH * outW;
            var tileHW = tileSize * tileSize;

            var outSpan = output.Memory.Span;
            var tileSpan = tile.Memory.Span;
            var weightSumSpan = weightSum.Memory.Span;

            var outX0 = posX * scaleFactor;
            var outY0 = posY * scaleFactor;

            var weightH = weight.GetLength(0);
            var weightW = weight.GetLength(1);

            Span<int> wxLut = stackalloc int[tileSize];
            Span<int> wyLut = stackalloc int[tileSize];
            for (int i = 0; i < tileSize; i++)
            {
                wxLut[i] = Math.Min(i / scaleFactor, weightW - 1);
                wyLut[i] = Math.Min(i / scaleFactor, weightH - 1);
            }

            for (int y = 0; y < tileSize; y++)
            {
                int oy = outY0 + y;
                if ((uint)oy >= (uint)outH)
                    continue;

                int wy = wyLut[y];
                int outRow = oy * outW;

                for (int x = 0; x < tileSize; x++)
                {
                    int ox = outX0 + x;
                    if ((uint)ox >= (uint)outW)
                        continue;

                    float w = weight[wy, wxLut[x]];
                    if (w == 0f)
                        continue;

                    int outIdx = outRow + ox;
                    int tileIdx = y * tileSize + x;

                    int o = outIdx;
                    int t = tileIdx;
                    for (int c = 0; c < channels; c++)
                    {
                        outSpan[o] += tileSpan[t] * w;
                        o += outHW;
                        t += tileHW;
                    }
                    weightSumSpan[outIdx] += w;
                }
            }
        }


        /// <summary>
        /// Normalizes the specified output.
        /// </summary>
        /// <param name="output">The output.</param>
        /// <param name="weightSum">The weight sum.</param>
        public static ImageTensor Normalize(ImageTensor output, Tensor<float> weightSum)
        {
            var height = output.Height;
            var width = output.Width;
            var channels = output.Dimensions[1];
            var outSpan = output.Memory.Span;
            var weightSpan = weightSum.Memory.Span;
            var hw = height * width;
            var channelStride = hw;
            for (int i = 0; i < hw; i++)
            {
                float w = weightSpan[i];
                if (w <= 0f)
                    continue;

                float inv = 1f / w;
                int baseIdx = i;
                for (int c = 0; c < channels; c++)
                {
                    outSpan[baseIdx] *= inv;
                    baseIdx += channelStride;
                }
            }
            return output;
        }

    }

    public record TileJob(int X, int Y, int TileSize);
}
