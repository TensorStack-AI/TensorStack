// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Vision;

namespace TensorStack.TextGeneration.Common
{
    public class GenerateResult
    {
        public int Beam { get; set; }
        public float Score { get; set; }
        public string Result { get; set; }
        public float PenaltyScore { get; set; }
        public List<CoordinateResult> CoordinateResults { get; set; }
    }
}
