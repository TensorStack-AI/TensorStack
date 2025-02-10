// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Vision;

namespace TensorStack.Florence.Common
{
    public record CoordinateResult
    {
        public string Label { get; set; }
        public CoordinateType CoordinateType { get; set; }
        public Coordinate<float>[] Coordinates { get; set; }
        public CoordinateBox<float> CoordinateBox { get; set; }
    }
}
