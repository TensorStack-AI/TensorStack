// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Transformers.Common
{
    public record SearchOptions : GenerateOptions
    {
        public SearchOptions(){}
        public SearchOptions(GenerateOptions options) : base(options) { }
    }
}
