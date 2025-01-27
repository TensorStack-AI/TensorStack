// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Numerics;

namespace TensorStack.Common.Vision
{
    public readonly record struct CoordinateBox<T> where T : INumber<T>
    {
        public CoordinateBox(T[] values)
        {
            if (values.Length == 4)
            {
                MinX = values[0];
                MinY = values[1];
                MaxX = values[2];
                MaxY = values[3];
            }
            else
            {
                MinX = values[0] < values[6] ? values[0] : values[6];
                MinY = values[1] < values[3] ? values[1] : values[3];
                MaxX = values[4] > values[2] ? values[4] : values[2];
                MaxY = values[5] > values[7] ? values[5] : values[7];
            }
        }

        public CoordinateBox(T minX, T minY, T maxX, T maxY)
            : this([minX, minY, maxX, maxY]) { }

        public T MinX { get; }
        public T MinY { get; }
        public T MaxX { get; }
        public T MaxY { get; }
    }
}
