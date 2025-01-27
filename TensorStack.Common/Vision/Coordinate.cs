// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Numerics;

namespace TensorStack.Common.Vision
{
    public readonly record struct Coordinate<T> where T : INumber<T>
    {
        public Coordinate(T[] values)
        {
            PosX = values[0];
            PosY = values[1];
        }

        public Coordinate(T posX, T posY)
        {
            PosX = posX;
            PosY = posY;
        }

        public T PosX { get; }
        public T PosY { get; }
    }
}
