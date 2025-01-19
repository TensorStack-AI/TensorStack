// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using TensorStack.Core.Inference;
using TensorStack.Common.Tensor;
using TensorStack.Common;

namespace TensorStack.Core
{
    /// <summary>
    /// Helper extensions for conversion from OrtValue to Tensor, TensorSpan
    /// </summary>
    public static class OrtExtensions
    {

        /// <summary>
        /// Creates an allocated output buffer on the device.
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>OrtValue.</returns>
        public static OrtValue CreateOutputBuffer(this NamedMetadata metadata, ReadOnlySpan<int> dimensions)
        {
            return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.Value.ElementDataType, dimensions.ToLong());
        }


        /// <summary>
        /// Span access to the OrtValue data
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>ReadOnlySpan&lt;T&gt;.</returns>
        public static ReadOnlySpan<T> AsSpan<T>(this OrtValue ortValue) where T : unmanaged
        {
            return ortValue.GetTensorDataAsSpan<T>();
        }


        /// <summary>
        /// ReadOnlySpan access to the OrtValue data
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>ReadOnlySpan&lt;T&gt;.</returns>
        public static ReadOnlySpan<T> AsReadOnlySpan<T>(this OrtValue ortValue) where T : unmanaged
        {
            return ortValue.GetTensorDataAsSpan<T>();
        }


        /// <summary>
        /// Create a view of the OrtValue as TensorSpan
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> AsTensorSpan(this OrtValue ortValue)
        {
            var dimensions = ortValue.GetDimensions();
            var typeInfo = ortValue.GetTensorTypeAndShape();
            return typeInfo.ElementDataType switch
            {
                // NOTE: Float16 & BFloat16 will cause a copy off device
                // These types dont have math functions so makes sense to copy to float for convienece
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => ortValue.ToTensor().AsTensorSpan(),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => ortValue.ToTensor().AsTensorSpan(),
                _ => new TensorSpan<float>(ortValue.GetTensorMutableDataAsSpan<float>(), dimensions)
            };
        }


        /// <summary>
        /// Create a view of the OrtValue as TensorSpan
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>TensorSpan&lt;T&gt;.</returns>
        public static TensorSpan<T> AsTensorSpan<T>(this OrtValue ortValue) where T : unmanaged
        {
            return new TensorSpan<T>(ortValue.GetTensorMutableDataAsSpan<T>(), ortValue.GetDimensions());
        }


        /// <summary>
        /// Copy TensorSpan data to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="metadata">The metadata.</param>
        /// <returns>OrtValue.</returns>
        public static OrtValue ToOrtValue<T>(this TensorSpan<T> tensor, NamedMetadata metadata) where T : unmanaged
        {
            return OrtValue.CreateTensorValueFromMemory<T>(OrtMemoryInfo.DefaultInstance, tensor.Span.ToArray(), tensor.Dimensions.ToLong());
        }


        /// <summary>
        /// Copy TensorSpan data to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="metadata">The metadata.</param>
        /// <returns>OrtValue.</returns>
        public static OrtValue ToOrtValue(this TensorSpan<string> tensor, NamedMetadata metadata)
        {
            return OrtValue.CreateFromStringTensor(new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<string>(tensor.Span.ToArray(), tensor.Dimensions));
        }


        /// <summary>
        /// Copy TensorSpan data to OrtValue.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="metadata">The metadata.</param>
        /// <returns>OrtValue.</returns>
        public static OrtValue ToOrtValue(this TensorSpan<float> tensor, NamedMetadata metadata)
        {
            var dimensions = tensor.Dimensions.ToLong();
            return metadata.Value.ElementDataType switch
            {
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int64 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToLongMemory(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToFloat16Memory(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToBFloat16Memory(), dimensions),
                _ => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToFloatMemory(), dimensions)
            };
        }


        /// <summary>
        /// Copy OrtValue data to float Tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public static Tensor<T> ToTensor<T>(this OrtValue ortValue) where T : unmanaged
        {
            return new Tensor<T>(ortValue.GetTensorDataAsSpan<T>().ToArray(), ortValue.GetDimensions());
        }


        /// <summary>
        /// Copy OrtValue data to float Tensor.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> ToTensor(this OrtValue ortValue)
        {
            var dimensions = ortValue.GetDimensions();
            var typeInfo = ortValue.GetTensorTypeAndShape();
            return typeInfo.ElementDataType switch
            {
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => new Tensor<float>(ortValue.GetTensorMutableDataAsSpan<Float16>().ToFloatMemory(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => new Tensor<float>(ortValue.GetTensorMutableDataAsSpan<BFloat16>().ToFloatMemory(), dimensions),
                _ => new Tensor<float>(ortValue.GetTensorDataAsSpan<float>().ToArray(), dimensions)
            };
        }


        /// <summary>
        /// Copy OrtValue data to array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>T[].</returns>
        public static T[] ToArray<T>(this OrtValue ortValue) where T : unmanaged
        {
            return ortValue.AsReadOnlySpan<T>().ToArray();
        }


        /// <summary>
        /// Copy OrtValue data to flot array.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>System.Single[].</returns>
        public static float[] ToArray(this OrtValue ortValue)
        {
            var typeInfo = ortValue.GetTensorTypeAndShape();
            return typeInfo.ElementDataType switch
            {
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => ortValue.GetTensorMutableDataAsSpan<Float16>().ToFloatMemory().ToArray(),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => ortValue.GetTensorMutableDataAsSpan<BFloat16>().ToFloatMemory().ToArray(),
                _ => ortValue.GetTensorDataAsSpan<float>().ToArray()
            };
        }


        /// <summary>
        /// Converts Optimization to GraphOptimizationLevel.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>GraphOptimizationLevel.</returns>
        public static GraphOptimizationLevel ToGraphOptimizationLevel(this Optimization configuration)
        {
            return configuration switch
            {
                Optimization.None => GraphOptimizationLevel.ORT_DISABLE_ALL,
                Optimization.Basic => GraphOptimizationLevel.ORT_ENABLE_BASIC,
                Optimization.Extended => GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                Optimization.All => GraphOptimizationLevel.ORT_ENABLE_ALL,
                _ => GraphOptimizationLevel.ORT_DISABLE_ALL,
            };
        }


        /// <summary>
        /// Gets the dimensions.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>System.Int32[].</returns>
        private static int[] GetDimensions(this OrtValue ortValue)
        {
            return ortValue.GetTensorTypeAndShape().Shape.ToInt();
        }


        /// <summary>
        /// Copy float Span to long Memory
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns>Memory&lt;System.Int64&gt;.</returns>
        private static Memory<long> ToLongMemory(this Span<float> inputMemory)
        {
            return Array.ConvertAll(inputMemory.ToArray(), Convert.ToInt64).AsMemory();
        }


        /// <summary>
        /// Copy float Span to float Memory .
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns>Memory&lt;System.Single&gt;.</returns>
        private static Memory<float> ToFloatMemory(this Span<float> inputMemory)
        {
            return inputMemory.ToArray().AsMemory();
        }


        /// <summary>
        /// Copy float Span to Float16 Memory.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns>Memory&lt;Float16&gt;.</returns>
        private static Memory<Float16> ToFloat16Memory(this Span<float> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = GC.AllocateUninitializedArray<Float16>(elementCount);
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (Float16)inputMemory[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Copy to float Span tp BFloat16 Memory.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns>Memory&lt;BFloat16&gt;.</returns>
        private static Memory<BFloat16> ToBFloat16Memory(this Span<float> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = GC.AllocateUninitializedArray<BFloat16>(elementCount);
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (BFloat16)inputMemory[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Copt to Float16 Span to float Memory.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns>Memory&lt;System.Single&gt;.</returns>
        private static Memory<float> ToFloatMemory(this Span<Float16> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = GC.AllocateUninitializedArray<float>(elementCount);
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (float)inputMemory[i];

            return floatArray.AsMemory();
        }


        /// <summary>
        /// Copy to BFloat16 Span to float Memory
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns>Memory&lt;System.Single&gt;.</returns>
        private static Memory<float> ToFloatMemory(this Span<BFloat16> inputMemory)
        {
            var elementCount = inputMemory.Length;
            var floatArray = GC.AllocateUninitializedArray<float>(elementCount);
            for (int i = 0; i < elementCount; i++)
                floatArray[i] = (float)inputMemory[i];

            return floatArray.AsMemory();
        }
    }
}
