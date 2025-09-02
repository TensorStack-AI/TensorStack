// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.TextGeneration.Tokenizers
{
    public record TokenizerConfig
    {
        public long BOS { get; set; } = 0;
        public long EOS { get; set; } = 1;
        public long UNK { get; set; } = 3;
        public long PAD { get; set; } = 1;
        public string Path { get; set; }
        public int MaxLength { get; set; } = 1024;
    }
}
