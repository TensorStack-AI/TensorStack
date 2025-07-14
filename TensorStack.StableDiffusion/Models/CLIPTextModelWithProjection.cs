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
    /// CLIPTextModelWithProjection: Frozen text-encoder [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
    /// </summary>
    public class CLIPTextModelWithProjection : CLIPTextModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CLIPTextModelWithProjection"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public CLIPTextModelWithProjection(CLIPModelConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Run the model inference with the specified token input
        /// </summary>
        /// <param name="tokenInput">The token input.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public override async Task<TextEncoderResult> RunAsync(TokenizerResult tokenInput, CancellationToken cancellationToken = default)
        {
            if (!this.IsLoaded())
                await LoadAsync(cancellationToken: cancellationToken);

            var paddedInput = PadOrTruncate(tokenInput);
            var hiddenStateCount = Metadata.Outputs.Count - 1;
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
                modelParameters.AddOutput([1, HiddenSize]);
                for (int i = 0; i < hiddenStateCount; i++)
                    modelParameters.AddOutput([1, SequenceLength, HiddenSize]);

                // Inference
                using (var results = await RunInferenceAsync(modelParameters))
                {
                    var promptEmbedsPooled = results[0].ToTensor();
                    var hiddenStates = new Tensor<float>[hiddenStateCount];
                    for (var i = 0; i < hiddenStateCount; i++)
                    {
                        using (var hiddenState = results.ElementAt(i + 1))
                            hiddenStates[i] = hiddenState.ToTensor();
                    }
                    return new TextEncoderResult(hiddenStates, promptEmbedsPooled);
                }
            }
        }

    }
}
