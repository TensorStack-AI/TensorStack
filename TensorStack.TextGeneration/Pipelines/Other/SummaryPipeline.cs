// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Other
{
    public class SummaryPipeline : EncoderDecoderPipeline
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SummaryPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public SummaryPipeline(SummaryConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Creates the Summary Pipeline
        /// </summary>
        /// <param name="modelPath">The model path.</param>
        /// <param name="provider">The provider.</param>
        /// <param name="tokenizerModel">The tokenizer model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <returns>SummaryPipeline.</returns>
        public static SummaryPipeline Create(string modelPath, ExecutionProvider provider, string tokenizerModel = "spiece.model", string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            var config = new SummaryConfig
            {
                Tokenizer = new T5Tokenizer( new TokenizerConfig
                {
                    BOS = 0,
                    EOS = 1,
                    Path = Path.Combine(modelPath, tokenizerModel)
                }),
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

            config.DecoderConfig.SetProvider(provider);
            config.EncoderConfig.SetProvider(provider);
            return new SummaryPipeline(config);
        }
    }
}