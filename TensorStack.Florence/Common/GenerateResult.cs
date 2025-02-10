// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;

namespace TensorStack.Florence.Common
{
    public class GenerateResult
    {
        public float Score { get; set; }
        public int BeamIndex { get; set; }
        public string TextResult { get; set; }
        public List<CoordinateResult> CoordinateResults { get; set; }
    }
}
