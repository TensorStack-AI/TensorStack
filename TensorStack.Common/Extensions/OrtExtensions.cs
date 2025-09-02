// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Numerics;
using TensorStack.Common.Tensor;
using OrtType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace TensorStack.Common
{
    /// <summary>
    /// Helper extensions for conversion from OrtValue to Tensor, TensorSpan
    /// </summary>
    public static class OrtExtensions
    {
        /// <summary>
        /// Creates a tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="tensor">The tensor value.</param>
        public static OrtValue CreateTensorOrtValue<T>(this NamedMetadata metadata, TensorSpan<T> tensor) where T : unmanaged, INumber<T>
        {
            return CreateOrtValue(metadata, tensor);
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
            return metadata.CreateTensorOrtValue(new TensorSpan<T>([value], [1]));
        }


        /// <summary>
        /// Creates a scalar tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="value">The value.</param>
        public static OrtValue CreateScalarOrtValue(this NamedMetadata metadata, string value)
        {
            return metadata.CreateTensorOrtValue(new TensorSpan<string>([value], [1]));
        }


        /// <summary>
        /// Creates a scalar tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="value">The value.</param>
        public static OrtValue CreateScalarOrtValue(this NamedMetadata metadata, bool value)
        {
            return metadata.CreateTensorOrtValue(new TensorSpan<bool>([value], [1]));
        }


        /// <summary>
        /// Creates a scalar tensor OrtValue.
        /// </summary>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="value">The value.</param>
        public static OrtValue CreateScalarOrtValue(this NamedMetadata metadata, byte value)
        {
            return metadata.CreateTensorOrtValue(new TensorSpan<byte>([value], [1]));
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
        /// <param name="ortValue">The ort value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> ToTensor(this OrtValue ortValue)
        {
            return CreateTensor<float>(ortValue);
        }


        /// <summary>
        /// Copy OrtValue data to float Tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public static Tensor<T> ToTensor<T>(this OrtValue ortValue) where T : unmanaged, INumber<T>
        {
            return CreateTensor<T>(ortValue);
        }


        /// <summary>
        /// Copy OrtValue data to flot array.
        /// </summary>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>System.Single[].</returns>
        public static float[] ToArray(this OrtValue ortValue)
        {
            return CreateArray<float>(ortValue);
        }


        /// <summary>
        /// Copy OrtValue data to array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>T[].</returns>
        public static T[] ToArray<T>(this OrtValue ortValue) where T : unmanaged, INumber<T>
        {
            return CreateArray<T>(ortValue);
        }


        /// <summary>
        /// Creates the Tensor from OrtValue.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The OrtValue.</param>
        private static Tensor<T> CreateTensor<T>(OrtValue ortValue) where T : unmanaged, INumber<T>
        {
            var metadata = ortValue.GetTensorTypeAndShape();
            var dimensions = metadata.Shape.ToInt();
            var buffer = CreateArray<T>(ortValue);
            return new Tensor<T>(buffer, dimensions);
        }


        /// <summary>
        /// Creates a OrtValue from Tensor
        /// </summary>
        /// <typeparam name="T">The type of input value</typeparam>
        /// <param name="metadata">The input metadata.</param>
        /// <param name="tensor">The tensor input.</param>
        private static OrtValue CreateOrtValue<T>(NamedMetadata metadata, TensorSpan<T> tensor) where T : unmanaged, INumber<T>
        {
            return CreateOrtValue(metadata.Value.ElementDataType, tensor);
        }


        /// <summary>
        /// Creates a OrtValue from Tensor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortType">Type of the ort.</param>
        /// <param name="tensor">The tensor.</param>
        /// <returns>OrtValue.</returns>
        public static OrtValue CreateOrtValue<T>(OrtType ortType, TensorSpan<T> tensor) where T : unmanaged, INumber<T>
        {
            var buffer = tensor.Span;
            var dimensions = tensor.Dimensions.ToLong();
            var memoryInstance = OrtMemoryInfo.DefaultInstance;
            return ortType switch
            {
                OrtType.Float => OrtValue.CreateTensorValueFromMemory<float>(memoryInstance, buffer.ConvertBuffer<T, float>(), dimensions),
                OrtType.UInt8 => OrtValue.CreateTensorValueFromMemory<byte>(memoryInstance, buffer.ConvertBuffer<T, byte>(), dimensions),
                OrtType.Int8 => OrtValue.CreateTensorValueFromMemory<sbyte>(memoryInstance, buffer.ConvertBuffer<T, sbyte>(), dimensions),
                OrtType.UInt16 => OrtValue.CreateTensorValueFromMemory<ushort>(memoryInstance, buffer.ConvertBuffer<T, ushort>(), dimensions),
                OrtType.Int16 => OrtValue.CreateTensorValueFromMemory<short>(memoryInstance, buffer.ConvertBuffer<T, short>(), dimensions),
                OrtType.Int32 => OrtValue.CreateTensorValueFromMemory<int>(memoryInstance, buffer.ConvertBuffer<T, int>(), dimensions),
                OrtType.Int64 => OrtValue.CreateTensorValueFromMemory<long>(memoryInstance, buffer.ConvertBuffer<T, long>(), dimensions),
                OrtType.Double => OrtValue.CreateTensorValueFromMemory<double>(memoryInstance, buffer.ConvertBuffer<T, double>(), dimensions),
                OrtType.UInt32 => OrtValue.CreateTensorValueFromMemory<uint>(memoryInstance, buffer.ConvertBuffer<T, uint>(), dimensions),
                OrtType.UInt64 => OrtValue.CreateTensorValueFromMemory<ulong>(memoryInstance, buffer.ConvertBuffer<T, ulong>(), dimensions),
                OrtType.Float16 => OrtValue.CreateTensorValueFromMemory<Float16>(memoryInstance, buffer.ConvertBufferFloat16(), dimensions),
                OrtType.BFloat16 => OrtValue.CreateTensorValueFromMemory<BFloat16>(memoryInstance, buffer.ConvertBufferBFloat16(), dimensions),
                _ => throw new NotImplementedException("Conversion is not currently implemented.")
            };
        }


        /// <summary>
        /// Clones the specified OrtValue.
        /// </summary>
        /// <param name="original">The original.</param>
        /// <returns>OrtValue.</returns>
        public static OrtValue Clone(this OrtValue original)
        {
            var info = original.GetTensorTypeAndShape();
            return info.ElementDataType switch
            {
                OrtType.Float => original.Clone<float>(info),
                OrtType.Float16 => original.Clone<Float16>(info),
                _ => throw new NotSupportedException($"Unsupported element type: {info.ElementDataType}")
            };
        }


        /// <summary>
        /// Clones the specified OrtValue.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="original">The original.</param>
        /// <param name="info">The information.</param>
        /// <returns>OrtValue.</returns>
        public static OrtValue Clone<T>(this OrtValue original, OrtTensorTypeAndShapeInfo info) where T : unmanaged
        {
            var newValue = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, info.ElementDataType, info.Shape);
            var source = original.GetTensorDataAsSpan<T>();
            var destination = newValue.GetTensorMutableDataAsSpan<T>();
            source.CopyTo(destination);
            return newValue;
        }


        /// <summary>
        /// Creates an Array from OrtValue.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The OrtValue.</param>
        private static T[] CreateArray<T>(OrtValue ortValue) where T : unmanaged, INumber<T>
        {
            var metadata = ortValue.GetTensorTypeAndShape();
            return metadata.ElementDataType switch
            {
                OrtType.Float => ortValue.ConvertBuffer<float, T>(),
                OrtType.UInt8 => ortValue.ConvertBuffer<byte, T>(),
                OrtType.Int8 => ortValue.ConvertBuffer<sbyte, T>(),
                OrtType.UInt16 => ortValue.ConvertBuffer<ushort, T>(),
                OrtType.Int16 => ortValue.ConvertBuffer<short, T>(),
                OrtType.Int32 => ortValue.ConvertBuffer<int, T>(),
                OrtType.Int64 => ortValue.ConvertBuffer<long, T>(),
                OrtType.Double => ortValue.ConvertBuffer<double, T>(),
                OrtType.UInt32 => ortValue.ConvertBuffer<uint, T>(),
                OrtType.UInt64 => ortValue.ConvertBuffer<ulong, T>(),
                OrtType.Float16 => ortValue.ConvertBufferFloat16<T>(),
                OrtType.BFloat16 => ortValue.ConvertBufferBFloat16<T>(),
                _ => throw new NotImplementedException("Conversion is not currently implemented.")
            };
        }


        /// <summary>
        /// Converts the buffer to INumber.
        /// </summary>
        /// <typeparam name="I"></typeparam>
        /// <typeparam name="O"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>O[].</returns>
        private static O[] ConvertBuffer<I, O>(this Span<I> input)
            where I : unmanaged, INumber<I>
            where O : unmanaged, INumber<O>
        {
            if (typeof(I) == typeof(O))
                return (O[])(object)input.ToArray();

            var result = GC.AllocateUninitializedArray<O>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = O.CreateSaturating(input[i]);

            return result;
        }


        /// <summary>
        /// Converts the buffer to INumber.
        /// </summary>
        /// <typeparam name="I"></typeparam>
        /// <typeparam name="O"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>O[].</returns>
        private static O[] ConvertBuffer<I, O>(this OrtValue ortValue)
              where I : unmanaged, INumberBase<I>
              where O : INumberBase<O>
        {

            var input = ortValue.GetTensorDataAsSpan<I>();
            if (typeof(I) == typeof(O))
                return (O[])(object)input.ToArray();

            var result = GC.AllocateUninitializedArray<O>(input.Length);
            System.Numerics.Tensors.TensorPrimitives.ConvertSaturating(input, result.AsSpan());
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
                return (Float16[])(object)input.ToArray();

            var result = GC.AllocateUninitializedArray<Float16>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = (Float16)float.CreateSaturating(input[i]);

            return result;
        }


        /// <summary>
        /// Converts the buffer Float16.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>T[].</returns>
        private static T[] ConvertBufferFloat16<T>(this OrtValue ortValue)
            where T : INumber<T>
        {
            var input = ortValue.GetTensorDataAsSpan<Float16>();
            var result = GC.AllocateUninitializedArray<T>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = T.CreateSaturating((float)input[i]);

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
                return (BFloat16[])(object)input.ToArray();

            var result = GC.AllocateUninitializedArray<BFloat16>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = (BFloat16)float.CreateSaturating(input[i]);

            return result;
        }


        /// <summary>
        /// Converts the buffer to BFloat16.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ortValue">The ort value.</param>
        /// <returns>T[].</returns>
        private static T[] ConvertBufferBFloat16<T>(this OrtValue ortValue)
            where T : INumber<T>
        {
            var input = ortValue.GetTensorDataAsSpan<BFloat16>();
            var result = GC.AllocateUninitializedArray<T>(input.Length);
            for (int i = 0; i < input.Length; i++)
                result[i] = T.CreateSaturating((float)input[i]);

            return result;
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
