// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Other
{
    public class SummaryPipeline : EncoderDecoderPipeline<GenerateOptions>, ITextGeneration
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SummaryPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public SummaryPipeline(SummaryConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Runs the GreedySearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;GenerateResult&gt; representing the asynchronous operation.</returns>
        public async Task<GenerateResult> RunAsync(GenerateOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);

            var sequence = await GreedySearchAsync(options, progressCallback, cancellationToken);
            using (sequence)
            {
                return new GenerateResult
                {
                    Score = sequence.Score,
                    PenaltyScore = sequence.PenaltyScore,
                    Result = Tokenizer.Decode(sequence.Tokens)
                };
            }
        }


        /// <summary>
        /// Runs the BeamSearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public async Task<GenerateResult[]> RunAsync(SearchOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);

            var sequences = await BeamSearchAsync(options, progressCallback, cancellationToken);
            var results = new GenerateResult[sequences.Length];
            for (int beam = 0; beam < sequences.Length; beam++)
            {
                var sequence = sequences[beam];
                using (sequence)
                {
                    results[beam] = new GenerateResult
                    {
                        Beam = beam,
                        Score = sequence.Score,
                        PenaltyScore = sequence.PenaltyScore,
                        Result = Tokenizer.Decode(sequence.Tokens)
                    };
                }
            }
            return results;
        }


        /// <summary>
        /// Creates the Summary Pipeline
        /// </summary>
        /// <param name="provider">The provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="tokenizerModel">The tokenizer model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <returns>SummaryPipeline.</returns>
        public static SummaryPipeline Create(ExecutionProvider provider, string modelPath, string tokenizerModel = "spiece.model", string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            return Create(provider, provider, modelPath, tokenizerModel, decoderModel, encoderModel);
        }


        /// <summary>
        /// Creates the Summary Pipeline
        /// </summary>
        /// <param name="encoderProvider">The encoder provider.</param>
        /// <param name="decoderProvider">The decoder provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="tokenizerModel">The tokenizer model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <returns>SummaryPipeline.</returns>
        public static SummaryPipeline Create(ExecutionProvider encoderProvider, ExecutionProvider decoderProvider, string modelPath, string tokenizerModel = "spiece.model", string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            var config = new SummaryConfig
            {
                Tokenizer = new T5Tokenizer(new TokenizerConfig
                {
                    BOS = 0,
                    EOS = 1,
                    Path = Path.Combine(modelPath, tokenizerModel)
                }),
                EncoderConfig = new EncoderConfig
                {
                    Path = Path.Combine(modelPath, encoderModel),
                    VocabSize = 32128,
                    NumHeads = 8,
                    NumLayers = 6,
                    HiddenSize = 512,
                },
                DecoderConfig = new DecoderConfig
                {
                    Path = Path.Combine(modelPath, decoderModel),
                    VocabSize = 32128,
                    NumHeads = 8,
                    NumLayers = 6,
                    HiddenSize = 512,
                }
            };

            config.EncoderConfig.SetProvider(encoderProvider);
            config.DecoderConfig.SetProvider(decoderProvider);
            return new SummaryPipeline(config);
        }
    }
}