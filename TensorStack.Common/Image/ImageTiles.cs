// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading.Tasks;
using TensorStack.Common.Tensor;

namespace TensorStack.Common.Image
{
    /// <summary>
    /// ImageTile Class to handle splitting and joining a larrgr image tensor into quarters with overlap.
    /// </summary>
    public record ImageTiles
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageTiles"/> class.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="overlap">The overlap.</param>
        /// <param name="tile1">The tile1.</param>
        /// <param name="tile2">The tile2.</param>
        /// <param name="tile3">The tile3.</param>
        /// <param name="tile4">The tile4.</param>
        public ImageTiles(int width, int height, int overlap, ImageTensor tile1, ImageTensor tile2, ImageTensor tile3, ImageTensor tile4)
        {
            Width = width;
            Height = height;
            Overlap = overlap;
            Tile1 = tile1;
            Tile2 = tile2;
            Tile3 = tile3;
            Tile4 = tile4;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageTiles"/> class.
        /// </summary>
        /// <param name="sourceTensor">The source tensor.</param>
        /// <param name="overlap">The overlap.</param>
        public ImageTiles(ImageTensor sourceTensor, int overlap = 16)
        {
            Overlap = overlap;
            Width = sourceTensor.Dimensions[3] / 2;
            Height = sourceTensor.Dimensions[2] / 2;
            Tile1 = SplitImageTile(sourceTensor, 0, 0, Height + overlap, Width + overlap);
            Tile2 = SplitImageTile(sourceTensor, 0, Width - overlap, Height + overlap, Width * 2);
            Tile3 = SplitImageTile(sourceTensor, Height - overlap, 0, Height * 2, Width + overlap);
            Tile4 = SplitImageTile(sourceTensor, Height - overlap, Width - overlap, Height * 2, Width * 2);
        }

        public int Width { get; init; }
        public int Height { get; init; }
        public int Overlap { get; init; }
        public ImageTensor Tile1 { get; init; }
        public ImageTensor Tile2 { get; init; }
        public ImageTensor Tile3 { get; init; }
        public ImageTensor Tile4 { get; init; }


        /// <summary>
        /// Joins the tiles into a single ImageTensor.
        /// </summary>
        /// <returns>ImageTensor.</returns>
        public ImageTensor JoinTiles()
        {
            var totalWidth = Width * 2;
            var totalHeight = Height * 2;
            var channels = Tile1.Dimensions[1];
            var destination = new ImageTensor(new[] { 1, channels, totalHeight, totalWidth });
            JoinTile(destination, Tile1, 0, 0, Height + Overlap, Width + Overlap);
            JoinTile(destination, Tile2, 0, Width - Overlap, Height + Overlap, totalWidth);
            JoinTile(destination, Tile3, Height - Overlap, 0, totalHeight, Width + Overlap);
            JoinTile(destination, Tile4, Height - Overlap, Width - Overlap, totalHeight, totalWidth);
            return destination;
        }


        /// <summary>
        /// Joins the tile to a destination.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="tile">The tile.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        private static void JoinTile(ImageTensor destination, ImageTensor tile, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = tile.Dimensions[1];
            Parallel.For(0, channels, (c) =>
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        var value = tile[0, c, i, j];
                        var existing = destination[0, c, startRow + i, startCol + j];
                        if (existing > 0)
                        {
                            value = (existing + value) / 2f;
                        }
                        destination[0, c, startRow + i, startCol + j] = value;
                    }
                }
            });
        }


        /// <summary>
        /// Splits the image tile.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="startRow">The start row.</param>
        /// <param name="startCol">The start col.</param>
        /// <param name="endRow">The end row.</param>
        /// <param name="endCol">The end col.</param>
        /// <returns>ImageTensor.</returns>
        private static ImageTensor SplitImageTile(ImageTensor source, int startRow, int startCol, int endRow, int endCol)
        {
            int height = endRow - startRow;
            int width = endCol - startCol;
            int channels = source.Dimensions[1];
            var splitTensor = new ImageTensor(new[] { 1, channels, height, width });
            Parallel.For(0, channels, (c) =>
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        splitTensor[0, c, i, j] = source[0, c, startRow + i, startCol + j];
                    }
                }
            });
            return splitTensor;
        }

    }
}
