// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.

using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Processing;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Phi
{
    public class Phi3Pipeline : DecoderPipeline, ITextGeneration
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Phi3Pipeline"/> class.
        /// </summary>
        /// <param name="tokenizerConfig">The tokenizer configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public Phi3Pipeline(Phi3Config configuration)
            : base(configuration.Tokenizer, configuration.DecoderConfig)
        {
            Configuration = configuration;
        }

        public Phi3Config Configuration { get; }


        /// <summary>
        /// Runs the GreedySearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public virtual async Task<GenerateResult> RunAsync(GenerateOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);
            var sequence = await GreedySearchAsync(options, cancellationToken);
            using (sequence)
            {
                return new GenerateResult
                {
                    Score = sequence.Score,
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
        public async Task<GenerateResult[]> RunAsync(SearchOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);

            var sequences = await BeamSearchAsync(options, cancellationToken);
            var results = new GenerateResult[sequences.Length];
            for (int i = 0; i < sequences.Length; i++)
            {
                var sequence = sequences[i];
                using (sequence)
                {
                    results[i] = new GenerateResult
                    {
                        Beam = sequence.Id,
                        Score = sequence.Score,
                        PenaltyScore = sequence.PenaltyScore,
                        Result = Tokenizer.Decode(sequence.Tokens)
                    };
                }
            }
            return results;
        }


        /// <summary>
        /// Initialize the Decoder cache
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task&lt;Sequence&gt; representing the asynchronous operation.</returns>
        protected override async Task<Sequence> InitializeAsync(GenerateOptions options)
        {
            var modelMetadata = await Decoder.LoadAsync();
            var dataType = modelMetadata.Outputs[0].Value.ElementDataType;
            var kvCache = new KVCacheDecoder(dataType, DecoderConfig.NumHeads, DecoderConfig.NumLayers, DecoderConfig.HiddenSize, DecoderConfig.NumKVHeads);
            var sequence = new Sequence(kvCache, Tokenizer.BOS);
            sequence.Initialize(TokenizerOutput.Length);

            var positionIds = GetPositionIds(modelMetadata, 0, TokenizerOutput.Length);
            var attentionMask = new Tensor<long>([1, TokenizerOutput.Length], 1);
            using (var parameters = new ModelParameters(modelMetadata))
            {
                // Inputs
                parameters.AddInput(TokenizerOutput.InputIds);
                if (positionIds != null)
                    parameters.AddInput(positionIds);
                parameters.AddInput(attentionMask);
                foreach (var pastKeyValue in sequence.Cache)
                    parameters.AddInput(pastKeyValue);

                // Outputs
                foreach (var output in modelMetadata.Outputs)
                    parameters.AddOutput();

                // Result
                var modelResult = Decoder.RunInference(parameters);
                using (var logitsResult = modelResult[0])
                {
                    var logits = logitsResult.ToTensor();
                    var presentKeyValues = modelResult.ToArray()[1..];
                    sequence.UpdateCache(presentKeyValues, false);
                }
            }
            return sequence;
        }


        /// <summary>
        /// Run decoder model
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <returns>A Task&lt;Tensor`1&gt; representing the asynchronous operation.</returns>
        protected override async Task<Tensor<float>> RunDecoderAsync(Sequence sequence)
        {
            var currentPosition = TokenizerOutput.Length + sequence.Tokens.Count;
            var modelMetadata = await Decoder.LoadAsync();
            var inputIds = new Tensor<long>([1, 1], sequence.Tokens[^1]);
            var positionIds = GetPositionIds(modelMetadata, currentPosition);
            var attentionMask = new Tensor<long>([1, currentPosition], 1);
            using (var parameters = new ModelParameters(modelMetadata))
            {
                // Inputs
                parameters.AddInput(inputIds);
                if (positionIds != null)
                    parameters.AddInput(positionIds);
                parameters.AddInput(attentionMask);
                foreach (var pastKeyValue in sequence.Cache)
                    parameters.AddInput(pastKeyValue, false);

                // Outputs
                foreach (var output in modelMetadata.Outputs)
                    parameters.AddOutput();

                // Result
                var modelResult = Decoder.RunInference(parameters);
                using (var logitsResult = modelResult[0])
                {
                    var logits = logitsResult.ToTensor();
                    var presentKeyValues = modelResult.ToArray()[1..];

                    sequence.UpdateCache(presentKeyValues, false);
                    return logits.Reshape([logits.Dimensions[0], logits.Dimensions[2]]);
                }
            }
        }


        /// <summary>
        /// Gets the token processors.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>ITokenProcessor[].</returns>
        protected override ITokenProcessor[] GetTokenProcessors(GenerateOptions options)
        {
            return
            [
                new EOSTokenProcessor
                (
                    options.MinLength, // min length
                    Tokenizer.EOS,
                    32000, // <|endoftext|>
                    32001 // <|assistant|> 
                   // 32007  // <|end|>
                ),
                new MaxLengthTokenProcessor(options.MaxLength)
            ];
        }


        /// <summary>
        /// Creates the Phi3Pipeline
        /// </summary>
        /// <param name="provider">The provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="tokenizerModel">The tokenizer model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <returns>Phi3Pipeline.</returns>
        public static Phi3Pipeline Create(ExecutionProvider provider, string modelPath, PhiType modelType, string tokenizerModel = "tokenizer.model", string decoderModel = "model.onnx")
        {
            var numHeads = 32;
            var numLayers = 32;
            var hiddenSize = 3072;
            var numKVHeads = 32;
            var vocabSize = 32064;
            if (modelType == PhiType.Mini)
            {
                numHeads = 32;
                numLayers = 32;
                hiddenSize = 3072;
                numKVHeads = 32;
            }
            else if (modelType == PhiType.Small)
            {
                numHeads = 32;
                numLayers = 32;
                hiddenSize = 4096;
                numKVHeads = 8;
            }
            else if (modelType == PhiType.Medium)
            {
                numHeads = 40;
                numLayers = 40;
                hiddenSize = 5120;
                numKVHeads = 10;
            }

            var config = new Phi3Config
            {
                Tokenizer = new T5Tokenizer(new TokenizerConfig
                {
                    BOS = 1,
                    EOS = 2,
                    Path = Path.Combine(modelPath, tokenizerModel)
                }),
                DecoderConfig = new DecoderConfig
                {
                    Path = Path.Combine(modelPath, decoderModel),
                    VocabSize = vocabSize,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    HiddenSize = hiddenSize,
                    NumKVHeads = numKVHeads
                }
            };

            config.DecoderConfig.SetProvider(provider);
            return new Phi3Pipeline(config);
        }

    }
}