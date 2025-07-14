// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    /// <summary>
    /// CLIPTextModel: Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))
    /// </summary>
    public class CLIPTextModel : ModelSession<CLIPModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CLIPTextModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public CLIPTextModel(CLIPModelConfig configuration)
            : base(configuration) { }

        /// <summary>
        /// Gets the pad token identifier.
        /// </summary>
        public int PadTokenId => Configuration.PadTokenId;

        /// <summary>
        /// Gets the hidden size.
        /// </summary>
        public int HiddenSize => Configuration.HiddenSize;

        /// <summary>
        /// Gets the sequence length.
        /// </summary>
        public int SequenceLength => Configuration.SequenceLength;

        /// <summary>
        /// Gets a value indicating whether this instance is fixed sequence length.
        /// </summary>
        public bool IsFixedSequenceLength => Configuration.IsFixedSequenceLength;

        /// <summary>
        /// Run the model inference with the specified token input
        /// </summary>
        /// <param name="tokenInput">The token input.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public virtual async Task<TextEncoderResult> RunAsync(TokenizerResult tokenInput, CancellationToken cancellationToken = default)
        {
            if (!this.IsLoaded())
                await LoadAsync(cancellationToken: cancellationToken);

            var paddedInput = PadOrTruncate(tokenInput);
            var supportsAttentionMask = Metadata.Inputs.Count == 2;
            var inputTensor = new TensorSpan<long>(paddedInput.InputIds, [1, SequenceLength]);
            var attentionTensor = new TensorSpan<long>(paddedInput.AttentionMask, [1, SequenceLength]);
            using (var modelParameters = new ModelParameters(Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(inputTensor);
                if (supportsAttentionMask)
                    modelParameters.AddInput(attentionTensor);

                // Outputs
                modelParameters.AddOutput([1, SequenceLength, HiddenSize]);
                modelParameters.AddOutput([1, HiddenSize]);

                // Inference
                var results = await RunInferenceAsync(modelParameters);
                using (var promptEmbeds = results.First())
                using (var promptPooledEmbeds = results.Last())
                {
                    return new TextEncoderResult(promptEmbeds.ToTensor(), promptPooledEmbeds.ToTensor());
                }
            }
        }


        /// <summary>
        /// Pads or truncate to fit SequenceLength.
        /// </summary>
        /// <param name="tokenizerResult">The tokenizer result.</param>
        /// <param name="padTokenId">The pad token identifier.</param>
        /// <param name="requiredLength">Length of the required.</param>
        /// <returns>TokenizerResult.</returns>
        protected TokenizerResult PadOrTruncate(TokenizerResult tokenizerResult)
        {
            var inputIds = tokenizerResult.InputIds.PadOrTruncate(PadTokenId, SequenceLength);
            var attentionMask = tokenizerResult.AttentionMask.PadOrTruncate(0, SequenceLength);
            var weights = tokenizerResult.Weights.PadOrTruncate(1, SequenceLength);
            return new TokenizerResult(inputIds, attentionMask, weights);
        }

    }
}
