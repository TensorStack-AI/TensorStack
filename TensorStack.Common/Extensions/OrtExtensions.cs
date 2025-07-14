// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Numerics;
using TensorStack.Common.Tensor;

namespace TensorStack.Common
{
    /// <summary>
    /// Helper extensions for conversion from OrtValue to Tensor, TensorSpan
    /// </summary>
    public static class OrtExtensions
    {
        public static OrtValue CreateTensorOrtValue<T>(this NamedMetadata metadata, TensorSpan<T> tensor) where T : unmanaged, INumber<T>
        {
            var buffer = tensor.Span;
            var dimensions = tensor.Dimensions.ToLong();
            return metadata.Value.ElementDataType switch
            {
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int32 => OrtValue.CreateTensorValueFromMemory<int>(OrtMemoryInfo.DefaultInstance, buffer.ConvertBuffer<T, int>(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int64 => OrtValue.CreateTensorValueFromMemory<long>(OrtMemoryInfo.DefaultInstance, buffer.ConvertBuffer<T, long>(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Double => OrtValue.CreateTensorValueFromMemory<double>(OrtMemoryInfo.DefaultInstance, buffer.ConvertBuffer<T, double>(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => OrtValue.CreateTensorValueFromMemory<Float16>(OrtMemoryInfo.DefaultInstance, buffer.ConvertBufferFloat16(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory<BFloat16>(OrtMemoryInfo.DefaultInstance, buffer.ConvertBufferBFloat16(), dimensions),
                _ => OrtValue.CreateTensorValueFromMemory<float>(OrtMemoryInfo.DefaultInstance, buffer.ConvertBuffer<T, float>(), dimensions)
            };
        }


        /// <summary>
        /// Creates a tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="tensor">The tensor value.</param>
        public static OrtValue CreateTensorOrtValue(this NamedMetadata metadata, TensorSpan<string> tensor)
        {
            return OrtValue.CreateFromStringTensor(new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<string>(new Memory<string>(tensor.Span.ToArray()), tensor.Dimensions));
        }


        /// <summary>
        /// Creates a tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="tensor">The tensor value.</param>
        public static OrtValue CreateTensorOrtValue(this NamedMetadata metadata, TensorSpan<bool> tensor)
        {
            return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<bool>(tensor.Span.ToArray()), tensor.Dimensions.ToLong());
        }


        /// <summary>
        /// Creates a tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="tensor">The tensor value.</param>
        public static OrtValue CreateTensorOrtValue(this NamedMetadata metadata, TensorSpan<byte> tensor)
        {
            return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<byte>(tensor.Span.ToArray()), tensor.Dimensions.ToLong());
        }


        /// <summary>
        /// Creates a scalar tensor OrtValue.
        /// </summary>
        /// <typeparam name="T">The type of input value</typeparam>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="value">The value.</param>
        public static OrtValue CreateScalarOrtValue<T>(this NamedMetadata metadata, T value) where T : unmanaged, INumber<T>
        {
            if (metadata.Value.ElementType == typeof(double))
                return metadata.CreateTensorOrtValue(new TensorSpan<double>(new[] { Convert.ToDouble(value) }, [1]));
            else if (metadata.Value.ElementType == typeof(int))
                return metadata.CreateTensorOrtValue(new TensorSpan<int>(new[] { Convert.ToInt32(value) }, [1]));
            else if (metadata.Value.ElementType == typeof(long))
                return metadata.CreateTensorOrtValue(new TensorSpan<long>(new[] { Convert.ToInt64(value) }, [1]));

            return metadata.CreateTensorOrtValue(new TensorSpan<float>(new[] { Convert.ToSingle(value) }, [1]));
        }


        /// <summary>
        /// Creates a scalar tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="value">The value.</param>
        public static OrtValue CreateScalarOrtValue(this NamedMetadata metadata, string value)
        {
            return metadata.CreateTensorOrtValue(new TensorSpan<string>(new[] { value }, [1]));
        }


        /// <summary>
        /// Creates a scalar tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="value">The value.</param>
        public static OrtValue CreateScalarOrtValue(this NamedMetadata metadata, bool value)
        {
            return metadata.CreateTensorOrtValue(new TensorSpan<bool>(new[] { value }, [1]));
        }


        /// <summary>
        /// Creates a scalar tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="value">The value.</param>
        public static OrtValue CreateScalarOrtValue(this NamedMetadata metadata, byte value)
        {
            return metadata.CreateTensorOrtValue(new TensorSpan<byte>(new[] { value }, [1]));
        }


        /// <summary>
        /// Creates and allocates the output tensor buffer.
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public static OrtValue CreateOutputBuffer(this NamedMetadata metadata, ReadOnlySpan<int> dimensions)
        {
            return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.Value.ElementDataType, dimensions.ToLong());
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
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => new Tensor<float>(ortValue.GetTensorDataAsSpan<Float16>().ToFloat(), dimensions),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => new Tensor<float>(ortValue.GetTensorDataAsSpan<BFloat16>().ToFloat(), dimensions),
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
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => ortValue.GetTensorDataAsSpan<Float16>().ToFloat(),
                Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => ortValue.GetTensorDataAsSpan<BFloat16>().ToFloat(),
                _ => ortValue.GetTensorDataAsSpan<float>().ToArray()
            };
        }




        private static O[] ConvertBuffer<I, O>(this Span<I> input)
            where I : unmanaged, INumber<I>
            where O : unmanaged, INumber<O>
        {
            if (typeof(I) == typeof(O))
            {
                //if (MemoryMarshal.TryGetArray(input, out ArraySegment<I> segment))
                //    return (O[])(object)segment.Array;

                return (O[])(object)input.ToArray();
            }

            var result = GC.AllocateUninitializedArray<O>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = O.CreateSaturating(input[i]);

            return result;
        }


        /// <summary>
        /// Converts the buffer to Float16.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="input">The input.</param>
        private static Float16[] ConvertBufferFloat16<T>(this Span<T> input)
            where T : unmanaged, INumber<T>
        {
            if (typeof(T) == typeof(Float16))
            {
                //if (MemoryMarshal.TryGetArray(input, out ArraySegment<T> segment))
                //    return (Float16[])(object)segment.Array;

                return (Float16[])(object)input.ToArray();
            }

            var result = GC.AllocateUninitializedArray<Float16>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = (Float16)float.CreateSaturating(input[i]);

            return result;
        }


        /// <summary>
        /// Converts the buffer to BFloat16.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="input">The input.</param>
        private static BFloat16[] ConvertBufferBFloat16<T>(this Span<T> input)
            where T : unmanaged, INumber<T>
        {
            if (typeof(T) == typeof(BFloat16))
            {
                //if (MemoryMarshal.TryGetArray(input, out ArraySegment<T> segment))
                //    return (BFloat16[])(object)segment.Array;

                return (BFloat16[])(object)input.ToArray();
            }

            var result = GC.AllocateUninitializedArray<BFloat16>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = (BFloat16)float.CreateSaturating(input[i]);

            return result;
        }


        /// <summary>
        /// Converts Float16 to float.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        private static float[] ToFloat(this ReadOnlySpan<Float16> input)
        {
            var result = GC.AllocateUninitializedArray<float>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = (float)input[i];

            return result;
        }


        /// <summary>
        /// Converts BFloat16 to float.
        /// </summary>
        /// <param name="inputMemory">The input memory.</param>
        /// <returns></returns>
        private static float[] ToFloat(this ReadOnlySpan<BFloat16> input)
        {
            var result = GC.AllocateUninitializedArray<float>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = (float)input[i];

            return result;
        }































        ///// <summary>
        ///// Creates an allocated output buffer on the device.
        ///// </summary>
        ///// <param name="metadata">The metadata.</param>
        ///// <param name="dimensions">The dimensions.</param>
        ///// <returns>OrtValue.</returns>
        //public static OrtValue CreateOutputBuffer(this NamedMetadata metadata, ReadOnlySpan<int> dimensions)
        //{
        //    return OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.Value.ElementDataType, dimensions.ToLong());
        //}


        ///// <summary>
        ///// Span access to the OrtValue data
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ortValue">The ort value.</param>
        ///// <returns>ReadOnlySpan&lt;T&gt;.</returns>
        //public static ReadOnlySpan<T> AsSpan<T>(this OrtValue ortValue) where T : unmanaged
        //{
        //    return ortValue.GetTensorDataAsSpan<T>();
        //}


        /// <summary>
        /// ReadOnlySpan access to the OrtValue data
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>ReadOnlySpan&lt;T&gt;.</returns>
        private static ReadOnlySpan<T> AsReadOnlySpan<T>(this OrtValue ortValue) where T : unmanaged
        {
            return ortValue.GetTensorDataAsSpan<T>();
        }


        ///// <summary>
        ///// Create a view of the OrtValue as TensorSpan
        ///// </summary>
        ///// <param name="ortValue">The ort value.</param>
        ///// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        //public static TensorSpan<float> AsTensorSpan(this OrtValue ortValue)
        //{
        //    var dimensions = ortValue.GetDimensions();
        //    var typeInfo = ortValue.GetTensorTypeAndShape();
        //    return typeInfo.ElementDataType switch
        //    {
        //        // NOTE: Float16 & BFloat16 will cause a copy off device
        //        // These types dont have math functions so makes sense to copy to float for convienece
        //        Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => ortValue.ToTensor().AsTensorSpan(),
        //        Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => ortValue.ToTensor().AsTensorSpan(),
        //        _ => new TensorSpan<float>(ortValue.GetTensorMutableDataAsSpan<float>(), dimensions)
        //    };
        //}


        ///// <summary>
        ///// Create a view of the OrtValue as TensorSpan
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        ///// <param name="ortValue">The ort value.</param>
        ///// <returns>TensorSpan&lt;T&gt;.</returns>
        //public static TensorSpan<T> AsTensorSpan<T>(this OrtValue ortValue) where T : unmanaged
        //{
        //    return new TensorSpan<T>(ortValue.GetTensorMutableDataAsSpan<T>(), ortValue.GetDimensions());
        //}


        ///// <summary>
        ///// Copy TensorSpan data to OrtValue.
        ///// </summary>
        ///// <param name="tensor">The tensor.</param>
        ///// <param name="metadata">The metadata.</param>
        ///// <returns>OrtValue.</returns>
        //public static OrtValue ToOrtValue<T>(this TensorSpan<T> tensor, NamedMetadata metadata) where T : unmanaged
        //{
        //    return OrtValue.CreateTensorValueFromMemory<T>(OrtMemoryInfo.DefaultInstance, tensor.Span.ToArray(), tensor.Dimensions.ToLong());
        //}


        ///// <summary>
        ///// Copy TensorSpan data to OrtValue.
        ///// </summary>
        ///// <param name="tensor">The tensor.</param>
        ///// <param name="metadata">The metadata.</param>
        ///// <returns>OrtValue.</returns>
        //public static OrtValue ToOrtValue(this TensorSpan<string> tensor, NamedMetadata metadata)
        //{
        //    return OrtValue.CreateFromStringTensor(new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<string>(tensor.Span.ToArray(), tensor.Dimensions));
        //}


        ///// <summary>
        ///// Copy TensorSpan data to OrtValue.
        ///// </summary>
        ///// <param name="tensor">The tensor.</param>
        ///// <param name="metadata">The metadata.</param>
        ///// <returns>OrtValue.</returns>
        //public static OrtValue ToOrtValue(this TensorSpan<float> tensor, NamedMetadata metadata)
        //{
        //    var dimensions = tensor.Dimensions.ToLong();
        //    return metadata.Value.ElementDataType switch
        //    {
        //        Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Int64 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToLongMemory(), dimensions),
        //        Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToFloat16Memory(), dimensions),
        //        Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToBFloat16Memory(), dimensions),
        //        _ => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Span.ToFloatMemory(), dimensions)
        //    };
        //}





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


        ///// <summary>
        ///// Copy float Span to long Memory
        ///// </summary>
        ///// <param name="inputMemory">The input memory.</param>
        ///// <returns>Memory&lt;System.Int64&gt;.</returns>
        //private static Memory<long> ToLongMemory(this Span<float> inputMemory)
        //{
        //    return Array.ConvertAll(inputMemory.ToArray(), Convert.ToInt64).AsMemory();
        //}


        ///// <summary>
        ///// Copy float Span to float Memory .
        ///// </summary>
        ///// <param name="inputMemory">The input memory.</param>
        ///// <returns>Memory&lt;System.Single&gt;.</returns>
        //private static Memory<float> ToFloatMemory(this Span<float> inputMemory)
        //{
        //    return inputMemory.ToArray().AsMemory();
        //}


        ///// <summary>
        ///// Copy float Span to Float16 Memory.
        ///// </summary>
        ///// <param name="inputMemory">The input memory.</param>
        ///// <returns>Memory&lt;Float16&gt;.</returns>
        //private static Memory<Float16> ToFloat16Memory(this Span<float> inputMemory)
        //{
        //    var elementCount = inputMemory.Length;
        //    var floatArray = GC.AllocateUninitializedArray<Float16>(elementCount);
        //    for (int i = 0; i < elementCount; i++)
        //        floatArray[i] = (Float16)inputMemory[i];

        //    return floatArray.AsMemory();
        //}


        ///// <summary>
        ///// Copy to float Span tp BFloat16 Memory.
        ///// </summary>
        ///// <param name="inputMemory">The input memory.</param>
        ///// <returns>Memory&lt;BFloat16&gt;.</returns>
        //private static Memory<BFloat16> ToBFloat16Memory(this Span<float> inputMemory)
        //{
        //    var elementCount = inputMemory.Length;
        //    var floatArray = GC.AllocateUninitializedArray<BFloat16>(elementCount);
        //    for (int i = 0; i < elementCount; i++)
        //        floatArray[i] = (BFloat16)inputMemory[i];

        //    return floatArray.AsMemory();
        //}


        ///// <summary>
        ///// Copt to Float16 Span to float Memory.
        ///// </summary>
        ///// <param name="inputMemory">The input memory.</param>
        ///// <returns>Memory&lt;System.Single&gt;.</returns>
        //private static Memory<float> ToFloatMemory(this Span<Float16> inputMemory)
        //{
        //    var elementCount = inputMemory.Length;
        //    var floatArray = GC.AllocateUninitializedArray<float>(elementCount);
        //    for (int i = 0; i < elementCount; i++)
        //        floatArray[i] = (float)inputMemory[i];

        //    return floatArray.AsMemory();
        //}


        ///// <summary>
        ///// Copy to BFloat16 Span to float Memory
        ///// </summary>
        ///// <param name="inputMemory">The input memory.</param>
        ///// <returns>Memory&lt;System.Single&gt;.</returns>
        //private static Memory<float> ToFloatMemory(this Span<BFloat16> inputMemory)
        //{
        //    var elementCount = inputMemory.Length;
        //    var floatArray = GC.AllocateUninitializedArray<float>(elementCount);
        //    for (int i = 0; i < elementCount; i++)
        //        floatArray[i] = (float)inputMemory[i];

        //    return floatArray.AsMemory();
        //}


        public static bool IsLoaded<T>(this ModelSession<T> session) where T : ModelConfig
        {
            if (session == null)
                return false;

            return session.Session is not null;
        }

        public static void CancelSession(this SessionOptions sessionOptions)
        {
            sessionOptions.SetLoadCancellationFlag(true);
        }


        public static void CancelSession(this RunOptions runOptions)
        {
            try
            {
                if (runOptions.IsClosed)
                    return;

                if (runOptions.IsInvalid)
                    return;

                if (runOptions.Terminate == true)
                    return;

                runOptions.Terminate = true;
            }
            catch (Exception)
            {
                throw new OperationCanceledException();
            }
        }
    }
}
