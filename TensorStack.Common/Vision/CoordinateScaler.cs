// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Tensor;

namespace TensorStack.Common.Vision
{
    /// <summary>
    /// CoordinateScaler class for scaling coordinates between model context size and image input size
    /// </summary>
    public class CoordinateScaler
    {
        private readonly int _contextWidth;
        private readonly int _contextHeight;

        /// <summary>
        /// Initializes a new instance of the <see cref="CoordinateScaler"/> class.
        /// </summary>
        /// <param name="contextWidth">Width of the context.</param>
        /// <param name="contextHeight">Height of the context.</param>
        public CoordinateScaler(int contextWidth, int contextHeight)
        {
            _contextWidth = contextWidth;
            _contextHeight = contextHeight;
        }


        /// <summary>
        /// Scales up the specified coordinates.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <returns>Coordinate&lt;System.Int32&gt;[].</returns>
        public Coordinate<int>[] ScaleDown(Coordinate<float>[] coordinates, ImageTensor sourceImage)
        {
            var scaleW = (float)sourceImage.Width / _contextWidth;
            var scaleH = (float)sourceImage.Height / _contextHeight;
            var scaledCoordinates = new Coordinate<int>[coordinates.Length];
            for (int i = 0; i < coordinates.Length; i++)
            {
                var posX = coordinates[i].PosX;
                var posY = coordinates[i].PosY;
                var scaledPosX = (int)Math.Clamp(MathF.Floor(posX / scaleW), 0, _contextWidth - 1);
                var scaledPosY = (int)Math.Clamp(MathF.Floor(posY / scaleH), 0, _contextHeight - 1);
                scaledCoordinates[i] = new Coordinate<int>(scaledPosX, scaledPosY);
            }
            return scaledCoordinates;
        }


        /// <summary>
        /// Scales down.
        /// </summary>
        /// <param name="boxes">The boxes.</param>
        /// <param name="coordinates">The coordinates.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <returns>CoordinateBox&lt;System.Int32&gt;[].</returns>
        public CoordinateBox<int>[] ScaleDown(CoordinateBox<float>[] boxes, ImageTensor sourceImage)
        {
            var scaleW = (float)sourceImage.Width / _contextWidth;
            var scaleH = (float)sourceImage.Height / _contextHeight;
            var coordinateBoxes = new CoordinateBox<int>[boxes.Length];
            for (var i = 0; i < boxes.Length; i++)
            {
                coordinateBoxes[i] = new CoordinateBox<int>(
                    minX: Math.Clamp((int)MathF.Floor(boxes[i].MinX / scaleW), 0, _contextWidth - 1),
                    minY: Math.Clamp((int)MathF.Floor(boxes[i].MinY / scaleH), 0, _contextHeight - 1),
                    maxX: Math.Clamp((int)MathF.Floor(boxes[i].MaxX / scaleW), 0, _contextWidth - 1),
                    maxY: Math.Clamp((int)MathF.Floor(boxes[i].MaxY / scaleH), 0, _contextHeight - 1));
            }
            return coordinateBoxes;
        }


        /// <summary>
        /// Scales up the specified coordinates.
        /// </summary>
        /// <param name="coordinates">The coordinates.</param>
        /// <param name="coordinates">The coordinates.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <returns>Coordinate&lt;System.Single&gt;[].</returns>
        public Coordinate<float>[] ScaleUp(Coordinate<int>[] coordinates, ImageTensor sourceImage)
        {
            var scaleW = (float)sourceImage.Width / _contextWidth;
            var scaleH = (float)sourceImage.Height / _contextHeight;
            var scaledCoordinates = new Coordinate<float>[coordinates.Length];
            for (var i = 0; i < coordinates.Length; i++)
            {
                var posX = coordinates[i].PosX;
                var posY = coordinates[i].PosY;
                var scaledPosX = (posX + 0.5f) * scaleW;
                var scaledPosY = (posY + 0.5f) * scaleH;
                scaledCoordinates[i] = new Coordinate<float>(scaledPosX, scaledPosY);
            }
            return scaledCoordinates;
        }


        /// <summary>
        /// Scales up the specified boxes.
        /// </summary>
        /// <param name="boxes">The boxes.</param>
        /// <param name="coordinates">The coordinates.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <returns>CoordinateBox&lt;System.Single&gt;[].</returns>
        public CoordinateBox<float>[] ScaleUp(CoordinateBox<int>[] boxes, ImageTensor sourceImage)
        {
            var scaleW = (float)sourceImage.Width / _contextWidth;
            var scaleH = (float)sourceImage.Height / _contextHeight;
            var coordinateBoxes = new CoordinateBox<float>[boxes.Length];
            for (var i = 0; i < boxes.Length; i++)
            {
                coordinateBoxes[i] = new CoordinateBox<float>(
                    minX: (boxes[i].MinX + 0.5f) * scaleW,
                    minY: (boxes[i].MinY + 0.5f) * scaleH,
                    maxX: (boxes[i].MaxX + 0.5f) * scaleW,
                    maxY: (boxes[i].MaxY + 0.5f) * scaleH);
            }
            return coordinateBoxes;
        }

    }
}