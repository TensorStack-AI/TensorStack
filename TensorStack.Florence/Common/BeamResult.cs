// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;

namespace TensorStack.Florence.Common
{
    public record BeamResult(int Index, List<long> Tokens)
    {
        public float Score { get; set; }
        public bool IsComplete { get; set; }
    }
}
