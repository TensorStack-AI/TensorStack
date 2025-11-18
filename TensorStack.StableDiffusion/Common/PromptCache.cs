using System;
using TensorStack.StableDiffusion.Pipelines;

namespace TensorStack.StableDiffusion.Common
{
    public record PromptCache
    {
        public string Conditional { get; init; }
        public string Unconditional { get; init; }
        public PromptResult CacheResult { get; init; }

        public bool IsValid(IPipelineOptions options)
        {
            return string.Equals(Conditional, options.Prompt, StringComparison.OrdinalIgnoreCase)
                && string.Equals(Unconditional, options.NegativePrompt, StringComparison.OrdinalIgnoreCase);
        }
    }
}
