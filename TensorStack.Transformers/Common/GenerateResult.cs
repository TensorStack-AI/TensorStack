// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Transformers.Common
{
    public class GenerateResult
    {
        public int Beam { get; set; }
        public float Score { get; set; }
        public string Result { get; set; }
    }
}
