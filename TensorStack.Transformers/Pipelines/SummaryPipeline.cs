// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.Transformers.Tokenizers;

namespace TensorStack.Transformers.Pipelines
{
    public class SummaryPipeline : TransformerPipeline
    {
        public SummaryPipeline(TransformerConfig configuration)
            : base(configuration) { }

        public static SummaryPipeline Create(string modelPath, ExecutionProvider provider, string tokenizerModel = "spiece.model", string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            var config = new TransformerConfig
            {
                TokenizerConfig = new T5TokenizerConfig
                {
                    BOS = 0,
                    EOS = 1,
                    Path = Path.Combine(modelPath, tokenizerModel)
                },
                DecoderConfig = new DecoderConfig
                {
                    Path = Path.Combine(modelPath, decoderModel),
                    VocabSize = 32128,
                    NumHeads = 8,
                    NumLayers = 6,
                    HiddenSize = 512,
                },
                EncoderConfig = new EncoderConfig
                {
                    Path = Path.Combine(modelPath, encoderModel),
                    VocabSize = 32128,
                    NumHeads = 8,
                    NumLayers = 6,
                    HiddenSize = 512,
                }
            };

            config.TokenizerConfig.SetProvider(provider);
            config.DecoderConfig.SetProvider(provider);
            config.EncoderConfig.SetProvider(provider);
            return new SummaryPipeline(config);
        }
    }
}