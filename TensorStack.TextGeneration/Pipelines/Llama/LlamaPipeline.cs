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
using TensorStack.TextGeneration.Cache;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Processing;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Llama
{
    public class LlamaPipeline : DecoderPipeline<GenerateOptions>,
        IPipeline<GenerateResult, GenerateOptions, GenerateProgress>,
        IPipeline<GenerateResult[], SearchOptions, GenerateProgress>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LlamaPipeline"/> class.
        /// </summary>
        /// <param name="tokenizerConfig">The tokenizer configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public LlamaPipeline(LlamaConfig configuration)
            : base(configuration.Tokenizer, configuration.DecoderConfig)
        {
            Configuration = configuration;
        }

        public LlamaConfig Configuration { get; }


        /// <summary>
        /// Runs the GreedySearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public virtual async Task<GenerateResult> RunAsync(GenerateOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);
            var sequence = await GreedySearchAsync(options, progressCallback, cancellationToken);
            using (sequence)
            {
                return new GenerateResult
                {
                    Score = sequence.Score,
                    Result = Tokenizer.Decode(sequence.Tokens),
                    Tokens = sequence.Tokens,
                    LastHiddenState = sequence.LastHiddenState
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
                        Result = Tokenizer.Decode(sequence.Tokens),
                        Tokens = sequence.Tokens,
                        LastHiddenState = sequence.LastHiddenState
                    };
                }
            }
            return results;
        }


        /// <summary>
        /// Gets the LastHiddenState.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> GetLastHiddenState(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);
            using (var sequence = await InitializeAsync(options))
            {
                return sequence.LastHiddenState;
            }
        }


        /// <summary>
        /// Tokenize the prompt
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        protected override async Task TokenizePromptAsync(GenerateOptions options)
        {
            var tokenizerResult = await Tokenizer.EncodeAsync(options.Prompt);
            var inputIds = tokenizerResult.InputIds.Span.Pad(Tokenizer.EOS, options.MinLength);
            var mask = tokenizerResult.Mask.Span.Pad(0, options.MinLength);
            TokenizerOutput = new TokenizerResult(inputIds, mask);
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
                    Tokenizer.EOS
                ),
                new MaxLengthTokenProcessor(options.MaxLength)
            ];
        }


        /// <summary>
        /// Initialize the Decoder cache
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task&lt;Sequence&gt; representing the asynchronous operation.</returns>
        protected override async Task<Sequence> InitializeAsync(GenerateOptions options)
        {
            var modelMetadata = await Decoder.LoadAsync();
            var kvCache = new KVCacheDecoder(modelMetadata, DecoderConfig.NumHeads, DecoderConfig.NumLayers, DecoderConfig.HiddenSize, DecoderConfig.NumKVHeads, options.MaxLength);
            var sequence = new Sequence(kvCache, Tokenizer.BOS);
            sequence.Initialize(0);

            var position = TokenizerOutput.Length;
            var inputIds = TokenizerOutput.InputIds;
            var positionIds = GetPositionIds(modelMetadata, 0, position);
            var attentionMask = new Tensor<long>([1, position], 1);
            RunDecoderInternal(modelMetadata, sequence, inputIds, positionIds, attentionMask, false);
            return sequence;
        }


        /// <summary>
        /// Run decoder model
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <returns>A Task&lt;Tensor`1&gt; representing the asynchronous operation.</returns>
        protected override async Task<Tensor<float>> RunDecoderAsync(Sequence sequence)
        {
            var modelMetadata = await Decoder.LoadAsync();
            var position = TokenizerOutput.Length + sequence.Tokens.Count;
            var inputIds = new Tensor<long>([1, 1], sequence.Tokens[^1]);
            var positionIds = GetPositionIds(modelMetadata, position);
            var attentionMask = new Tensor<long>([1, position], 1);
            return RunDecoderInternal(modelMetadata, sequence, inputIds, positionIds, attentionMask, true);
        }


        /// <summary>
        /// Runs the decoder
        /// </summary>
        /// <param name="modelMetadata">The model metadata.</param>
        /// <param name="sequence">The sequence.</param>
        /// <param name="inputIds">The input ids.</param>
        /// <param name="positionIds">The position ids.</param>
        /// <param name="attentionMask">The attention mask.</param>
        /// <param name="useBranchCache">if set to <c>true</c> [use branch cache].</param>
        private Tensor<float> RunDecoderInternal(ModelMetadata modelMetadata, Sequence sequence, Tensor<long> inputIds, Tensor<long> positionIds, Tensor<long> attentionMask, bool useBranchCache)
        {
            using (var parameters = new ModelParameters(modelMetadata))
            {
                // Inputs
                parameters.AddInput(inputIds);
                parameters.AddInput(attentionMask);
                if (positionIds != null)
                    parameters.AddInput(positionIds);

                foreach (var pastKeyValue in sequence.Cache)
                    parameters.AddInput(pastKeyValue, false);

                // Outputs
                foreach (var output in modelMetadata.Outputs)
                    parameters.AddOutput();

                // Result
                var modelResult = Decoder.RunInference(parameters);
                using (var logitsResult = modelResult[0])
                {
                    var dimension = logitsResult.GetDimensions();
                    var logits = logitsResult.ToTensor(dimension[1..]);
                    var lastHiddenState = Configuration.OutputLastHiddenStates ? modelResult[^1].ToTensor() : default;
                    var presentKeyValues = Configuration.OutputLastHiddenStates ? modelResult.ToArray()[1..^1] : modelResult.ToArray()[1..];
                    sequence.UpdateCache(presentKeyValues, useBranchCache, lastHiddenState);
                    return logits;
                }
            }
        }


        /// <summary>
        /// Creates the LlamaPipeline
        /// </summary>
        /// <param name="provider">The provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="tokenizerModel">The tokenizer model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <returns>Phi3Pipeline.</returns>
        public static LlamaPipeline Create(ExecutionProvider provider, string modelPath, string model = "model.onnx")
        {
            // Llama-3.2-1B
            var numHeads = 32;
            var numLayers = 16;
            var hiddenSize = 2048;
            var numKVHeads = 8;
            var vocabSize = 128256;
            var config = new LlamaConfig
            {
                OutputLastHiddenStates = true,
                Tokenizer = new BPETokenizer(new TokenizerConfig
                {
                    BOS = 128000,
                    EOS = 128001,
                    Path = modelPath
                }),
                DecoderConfig = new DecoderConfig
                {
                    Path = Path.Combine(modelPath, model),
                    VocabSize = vocabSize,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    HiddenSize = hiddenSize,
                    NumKVHeads = numKVHeads
                }
            };

            config.DecoderConfig.SetProvider(provider);
            return new LlamaPipeline(config);
        }

    }
}