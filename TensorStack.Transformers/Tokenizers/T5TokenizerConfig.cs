// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace TensorStack.Transformers.Tokenizers
{
    public record T5TokenizerConfig : ModelConfig
    {
        public long BOS { get; set; } = 0;
        public long EOS { get; set; } = 1;
    }
}
