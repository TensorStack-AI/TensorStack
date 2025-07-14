// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    /// <summary>
    /// T5EncoderModel: Frozen text-encoder [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl).
    /// </summary>
    public class T5EncoderModel : CLIPTextModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="T5EncoderModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public T5EncoderModel(CLIPModelConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Run the model inference with the specified token input
        /// </summary>
        /// <param name="tokenInput">The token input.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A Task&lt;TextEncoderResult&gt; representing the asynchronous operation.</returns>
        public override async Task<TextEncoderResult> RunAsync(TokenizerResult tokenInput, CancellationToken cancellationToken = default)
        {
            if (!this.IsLoaded())
                await LoadAsync(cancellationToken: cancellationToken);

            if (IsFixedSequenceLength)
                tokenInput = PadOrTruncate(tokenInput);

            var sequenceLength = tokenInput.InputIds.Length;
            var supportsAttentionMask = Metadata.Outputs.Count == 2;
            var inputTensor = new TensorSpan<long>(tokenInput.InputIds, [1, sequenceLength]);
            var attentionTensor = new TensorSpan<long>(tokenInput.AttentionMask, [1, sequenceLength]);
            using (var modelParameters = new ModelParameters(Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(inputTensor);
                if (supportsAttentionMask)
                    modelParameters.AddInput(attentionTensor);

                // Outputs
                modelParameters.AddOutput([1, sequenceLength, HiddenSize]);

                // Inference
                using (var results = await RunInferenceAsync(modelParameters))
                {
                    return new TextEncoderResult(results[0].ToTensor(), default);
                }
            }
        }

    }
}
