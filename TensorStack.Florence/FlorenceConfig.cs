// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.Florence
{
    public record FlorenceConfig
    {
        public string Path { get; set; }
        public int ImageSampleSize { get; set; } = 768;
        public int ImageSeqLength { get; set; } = 577;
        public int ImageContextWidth { get; set; } = 1000;
        public int ImageContextHeight { get; set; } = 1000;
        public int NumDecoderLayers { get; set; }
        public int NumDecoderHeads { get; set; }
        public int DecoderHiddenSize { get; set; }
        public int NumEncoderLayers { get; set; }
        public int NumEncoderHeads { get; set; }
        public int EncoderHiddenSize { get; set; }


        public static FlorenceConfig DefaultBase = new FlorenceConfig()
        {
            NumDecoderLayers = 6,
            NumDecoderHeads = 12,
            DecoderHiddenSize = 768,
            NumEncoderLayers = 6,
            NumEncoderHeads = 12,
            EncoderHiddenSize = 768
        };


        public static FlorenceConfig DefaultLarge = new FlorenceConfig()
        {
            NumDecoderLayers = 12,
            NumDecoderHeads = 16,
            DecoderHiddenSize = 1024,
            NumEncoderLayers = 12,
            NumEncoderHeads = 16,
            EncoderHiddenSize = 1024
        };
    }
}
