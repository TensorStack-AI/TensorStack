using TensorStack.Common.Pipeline;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration
{
    public interface ITextGeneration :
        IPipeline<GenerateResult, GenerateOptions>,
        IPipeline<GenerateResult[], SearchOptions>
    {
    }
}
