using CSnakes.Runtime.Python;
using System;
using TensorStack.Common.Tensor;
using TensorStack.Python.Common;

namespace TensorStack.Python
{
    internal static class Extensions
    {
        /// <summary>
        /// IPyBuffer to Tensor<float>.
        /// </summary>
        /// <param name="pyBuffer">The py buffer.</param>
        public static Tensor<float> ToTensor(this IPyBuffer pyBuffer)
        {
            var buffer = pyBuffer.GetBuffer();
            var dimensions = pyBuffer.GetDimensions();
            return new Tensor<float>(buffer, dimensions);
        }


        /// <summary>
        /// Gets the IPyBuffer buffer.
        /// </summary>
        /// <param name="pyBuffer">The py buffer.</param>
        public static Memory<float> GetBuffer(this IPyBuffer pyBuffer)
        {
            return pyBuffer.AsReadOnlySpan<float>().ToArray();
        }


        /// <summary>
        /// Gets the IPyBuffer dimensions.
        /// </summary>
        /// <param name="pyBuffer">The py buffer.</param>
        public static ReadOnlySpan<int> GetDimensions(this IPyBuffer pyBuffer)
        {
            var shape = pyBuffer.Shape;
            var dimensions = new int[shape.Length];
            for (int i = 0; i < dimensions.Length; i++)
            {
                dimensions[i] = (int)shape[i];
            }
            return dimensions;
        }


        public static void SendMessage(this IProgress<PipelineProgress> progressCallback, string message)
        {
            progressCallback?.Report(new PipelineProgress
            {
                Message = message,
                Process = "Initialize"
            });
        }
    }
}
