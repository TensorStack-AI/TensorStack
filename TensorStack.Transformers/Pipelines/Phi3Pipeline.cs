// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Transformers.Common;
using TensorStack.Transformers.Processing;
using TensorStack.Transformers.Tokenizers;

namespace TensorStack.Transformers.Pipelines
{
    public class Phi3Pipeline : PipelineBase,
        IPipeline<GenerateResult, GenerateOptions>,
        IPipelineStream<GenerateResult, SearchOptions>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Phi3Pipeline"/> class.
        /// </summary>
        /// <param name="tokenizerConfig">The tokenizer configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public Phi3Pipeline(Phi3Config configuration)
            : base(configuration.TokenizerConfig, configuration.DecoderConfig)
        {
            Configuration = configuration;
        }

        public Phi3Config Configuration { get; }

        /// <summary>
        /// Runs the model inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public virtual async Task<GenerateResult> RunAsync(GenerateOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return await RunAsync(new SearchOptions(options), progressCallback, cancellationToken).FirstAsync(cancellationToken);
        }


        public virtual async IAsyncEnumerable<GenerateResult> RunAsync(SearchOptions options, IProgress<RunProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            TokenizerOutput = await Tokenizer.EncodeAsync(options.Prompt);
            var sequences = await BeamSearchAsync(options, cancellationToken);
            foreach ( var sequence in sequences)
            {
                using (sequence)
                {
                    yield return new GenerateResult
                    {
                        Beam = sequence.Id,
                        Score = sequence.Score,
                        Result = Tokenizer.Decode(sequence.Tokens)
                    };
                }
            }
        }


        protected override async Task<Sequence> InitializeAsync(GenerateOptions options)
        {
            var modelMetadata = await Decoder.LoadAsync();
            var dataType = modelMetadata.Outputs[0].Value.ElementDataType;
            var kvCache = new KVCacheDecoder(dataType, DecoderConfig.NumHeads, DecoderConfig.NumLayers, DecoderConfig.HiddenSize);
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


        protected override ITokenProcessor[] GetTokenProcessors(GenerateOptions options)
        {
            return
            [
                new EOSTokenProcessor
                (
                    20, // min length
                    Tokenizer.EOS,
                    32000, // <|endoftext|>
                    32001, // <|assistant|> 
                    32007  // <|end|>
                ),
                new MaxLengthTokenProcessor(options.MaxLength)
            ];
        }


        public static Phi3Pipeline Create(string modelPath, ExecutionProvider provider, string tokenizerModel = "tokenizer.model", string decoderModel = "model.onnx")
        {
            var config = new Phi3Config
            {
                TokenizerConfig = new T5TokenizerConfig
                {
                    BOS = 1,
                    EOS = 2,
                    Path = Path.Combine(modelPath, tokenizerModel)
                },
                DecoderConfig = new DecoderConfig
                {
                    Path = Path.Combine(modelPath, decoderModel),
                    VocabSize = 32064,
                    NumHeads = 32,
                    NumLayers = 32,
                    HiddenSize = 3072,
                }
            };

            config.TokenizerConfig.SetProvider(provider);
            config.DecoderConfig.SetProvider(provider);
            return new Phi3Pipeline(config);
        }

    }
}