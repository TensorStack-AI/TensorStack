// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using TensorStack.Common.Tensor;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace TensorStack.Common
{
    /// <summary>
    /// Helper extensions for Tensor and TensorSpan, Math, Copy etc.
    /// </summary>
    public static class TensorExtensions
    {

        /// <summary>
        /// Divides the specified value from all tensor values.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> Divide(this TensorSpan<float> tensor, float value)
        {
            TensorPrimitives.Divide(tensor.Span, value, tensor.Span);
            return tensor;
        }


        /// <summary>
        /// Multiplies each Tensor value by the specified value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> Multiply(this TensorSpan<float> tensor, float value)
        {
            TensorPrimitives.Multiply(tensor.Span, value, tensor.Span);
            return tensor;
        }


        /// <summary>
        /// Adds TensorB to tensorA
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> Add(this TensorSpan<float> tensorA, TensorSpan<float> tensorB)
        {
            TensorPrimitives.Add(tensorA.Span, tensorB.Span, tensorA.Span);
            return tensorA;
        }


        /// <summary>
        /// Adds the specified value to each Tensor value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> Add(this TensorSpan<float> tensor, float value)
        {
            TensorPrimitives.Add(tensor.Span, value, tensor.Span);
            return tensor;
        }


        /// <summary>
        /// Subtracts TensorB from TensorA
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> Subtract(this TensorSpan<float> tensorA, TensorSpan<float> tensorB)
        {
            TensorPrimitives.Subtract(tensorA.Span, tensorB.Span, tensorA.Span);
            return tensorA;
        }


        /// <summary>
        /// Subtracts the specified value from each Tensor value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> Subtract(this TensorSpan<float> tensor, float value)
        {
            TensorPrimitives.Subtract(tensor.Span, value, tensor.Span);
            return tensor;
        }


        /// <summary>
        /// Divides the specified value from all tensor values.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <param name="isCopy">if set to <c>true</c> copy result to new tensor, othewise tensor is mutated</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Divide(this Tensor<float> tensor, float value, bool isCopy = false)
        {
            var result = isCopy ? new Tensor<float>(tensor.Dimensions) : tensor;
            TensorPrimitives.Divide(tensor.Span, value, result.Memory.Span);
            return result;
        }


        /// <summary>
        /// Multiplies each Tensor value by the specified value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <param name="isCopy">if set to <c>true</c> copy result to new tensor, othewise tensor is mutated</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Multiply(this Tensor<float> tensor, float value, bool isCopy = false)
        {
            var result = isCopy ? new Tensor<float>(tensor.Dimensions) : tensor;
            TensorPrimitives.Multiply(tensor.Span, value, result.Memory.Span);
            return result;
        }


        /// <summary>
        /// Adds TensorB to tensorA
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <param name="isCopy">if set to <c>true</c> copy result to new tensor, othewise tensor is mutated</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Add(this Tensor<float> tensorA, Tensor<float> tensorB, bool isCopy = false)
        {
            var result = isCopy ? new Tensor<float>(tensorA.Dimensions) : tensorA;
            TensorPrimitives.Add(tensorA.Span, tensorB.Span, result.Memory.Span);
            return result;
        }


        /// <summary>
        /// Adds the specified value to each Tensor value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <param name="isCopy">if set to <c>true</c> copy result to new tensor, othewise tensor is mutated</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Add(this Tensor<float> tensor, float value, bool isCopy = false)
        {
            var result = isCopy ? new Tensor<float>(tensor.Dimensions) : tensor;
            TensorPrimitives.Add(tensor.Span, value, result.Memory.Span);
            return result;
        }


        /// <summary>
        /// Subtracts TensorB from TensorA
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <param name="isCopy">if set to <c>true</c> copy result to new tensor, othewise tensor is mutated</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Subtract(this Tensor<float> tensorA, Tensor<float> tensorB, bool isCopy = false)
        {
            var result = isCopy ? new Tensor<float>(tensorA.Dimensions) : tensorA;
            TensorPrimitives.Subtract(tensorA.Span, tensorB.Span, result.Memory.Span);
            return result;
        }


        /// <summary>
        /// Subtracts the specified value from each Tensor value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <param name="isCopy">if set to <c>true</c> copy result to new tensor, othewise tensor is mutated</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Subtract(this Tensor<float> tensor, float value, bool isCopy = false)
        {
            var result = isCopy ? new Tensor<float>(tensor.Dimensions) : tensor;
            TensorPrimitives.Subtract(tensor.Span, value, result.Memory.Span);
            return result;
        }

        /// <summary>
        /// Reshapes the Tensor with the specified dimensions.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <param name="isCopy">if set to <c>true</c> copy result to new tensor, othewise tensor is mutated</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Reshape(this Tensor<float> tensor, ReadOnlySpan<int> dimensions, bool isCopy = false)
        {
            if (isCopy)
                return new Tensor<float>(tensor.Memory.ToArray(), dimensions);

            tensor.ReshapeTensor(dimensions);
            return tensor;
        }


        /// <summary>
        /// Copy TensorSpan to Tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public static Tensor<T> ToTensor<T>(this TensorSpan<T> tensor)
        {
            return new Tensor<T>(tensor.Span.ToArray(), tensor.Dimensions);
        }


        /// <summary>
        /// Copy Tensor to TensorSpan.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor">The tensor.</param>
        /// <returns>TensorSpan&lt;T&gt;.</returns>
        public static TensorSpan<T> ToTensorSpan<T>(this Tensor<T> tensor)
        {
            return new TensorSpan<T>(tensor.Memory.Span.ToArray(), tensor.Dimensions);
        }


        /// <summary>
        /// Copy TensorSpan to TensorSpan.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor">The tensor.</param>
        /// <returns>TensorSpan&lt;T&gt;.</returns>
        public static TensorSpan<T> ToTensorSpan<T>(this TensorSpan<T> tensor)
        {
            return new TensorSpan<T>(tensor.Span.ToArray(), tensor.Dimensions);
        }


        /// <summary>
        /// TensorSpan view of the ImageTensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageTensor ToImageTensor(this TensorSpan<float> tensor)
        {
            return tensor.ToTensor().AsImageTensor();
        }


        /// <summary>
        /// VideoTensor view of the TensorSpan.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="framerate">The framerate.</param>
        /// <returns>VideoTensor.</returns>
        public static VideoTensor ToVideoTensor(this TensorSpan<float> tensor, float framerate)
        {
            return tensor.ToTensor().AsVideoTensor(framerate);
        }


        /// <summary>
        /// TensorSpan view of the Tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor">The tensor.</param>
        /// <returns>TensorSpan&lt;T&gt;.</returns>
        public static TensorSpan<T> AsTensorSpan<T>(this Tensor<T> tensor)
        {
            return new TensorSpan<T>(tensor.Memory.Span, tensor.Dimensions);
        }


        /// <summary>
        /// ImageTensor view of the Tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageTensor AsImageTensor(this Tensor<float> tensor)
        {
            return new ImageTensor(tensor);
        }


        /// <summary>
        /// VideoTensor view of the Tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="framerate">The framerate.</param>
        /// <returns>VideoTensor.</returns>
        public static VideoTensor AsVideoTensor(this Tensor<float> tensor, float framerate)
        {
            return new VideoTensor(tensor, framerate);
        }


        /// <summary>
        /// Repeats the specified Tensor across axis 0.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor">The tensor.</param>
        /// <param name="count">The count.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        /// <exception cref="NotImplementedException">Only axis 0 is supported</exception>
        public static Tensor<T> Repeat<T>(this Tensor<T> tensor, int count, int axis = 0)
        {
            if (count == 1)
                return tensor;

            if (axis != 0)
                throw new NotImplementedException("Only axis 0 is supported");

            var dimensions = tensor.Dimensions.ToArray();
            dimensions[0] *= count;

            var length = (int)tensor.Length;
            var totalLength = length * count;
            var buffer = new T[totalLength].AsMemory();
            for (int i = 0; i < count; i++)
            {
                tensor.Memory.CopyTo(buffer[(i * length)..]);
            }
            return new Tensor<T>(buffer, dimensions);
        }


        /// <summary>
        /// Permutes the specified Tensor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor">The tensor.</param>
        /// <param name="permutation">The permutation.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public static Tensor<T> Permute<T>(this Tensor<T> tensor, int[] permutation)
        {
            var dimensions = tensor.Dimensions.ToArray();
            var newDimensions = permutation.Select(i => dimensions[i]).ToArray();
            var resultTensor = new Tensor<T>(newDimensions);
            var originalIndex = new int[dimensions.Length];
            var permutedIndex = new int[newDimensions.Length];

            for (int i = 0; i < tensor.Length; i++)
            {
                int remaining = i;
                for (int j = dimensions.Length - 1; j >= 0; j--)
                {
                    originalIndex[j] = remaining % dimensions[j];
                    remaining /= dimensions[j];
                }

                for (int j = 0; j < newDimensions.Length; j++)
                {
                    permutedIndex[j] = originalIndex[permutation[j]];
                }

                var multiplier = 1;
                var permutedFlatIndex = 0;
                for (int j = newDimensions.Length - 1; j >= 0; j--)
                {
                    permutedFlatIndex += permutedIndex[j] * multiplier;
                    multiplier *= newDimensions[j];
                }

                resultTensor.Memory.Span[permutedFlatIndex] = tensor.Memory.Span[i];
            }
            return resultTensor;
        }


        /// <summary>
        /// Splits the specified Tensors across axis 0.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>IEnumerable&lt;Tensor&lt;System.Single&gt;&gt;.</returns>
        /// <exception cref="NotImplementedException">Only axis 0 is supported</exception>
        public static IEnumerable<Tensor<float>> Split(this Tensor<float> tensor, int axis = 0)
        {
            if (axis != 0)
                throw new NotImplementedException("Only axis 0 is supported");

            var count = tensor.Dimensions[0];
            var dimensions = tensor.Dimensions.ToArray();
            dimensions[0] = 1;

            var newLength = (int)tensor.Length / count;
            for (int i = 0; i < count; i++)
            {
                var start = i * newLength;
                yield return new Tensor<float>(tensor.Memory.Slice(start, newLength), dimensions);
            }
        }


        /// <summary>
        /// Joins the specified Tensors across axis 0.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        /// <exception cref="NotImplementedException">Only axis 0 is supported</exception>
        public static Tensor<float> Join(this IEnumerable<Tensor<float>> tensors, int axis = 0)
        {
            if (axis != 0)
                throw new NotImplementedException("Only axis 0 is supported");

            var count = tensors.Count();
            var tensor = tensors.First();
            var dimensions = tensor.Dimensions.ToArray();
            dimensions[0] *= count;

            var newLength = (int)tensor.Length;
            var buffer = new float[newLength * count].AsMemory();

            var index = 0;
            foreach (var item in tensors)
            {
                var start = index * newLength;
                item.Memory.CopyTo(buffer[start..]);
                index++;
            }
            return new Tensor<float>(buffer, dimensions);
        }


        /// <summary>
        /// Generates the next random tensor
        /// </summary>
        /// <param name="random">The random.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <param name="initNoiseSigma">The initialize noise sigma.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> NextTensor(this Random random, ReadOnlySpan<int> dimensions, float initNoiseSigma = 1f)
        {
            var latents = new Tensor<float>(dimensions);
            for (int i = 0; i < latents.Length; i++)
            {
                var u1 = random.NextSingle();
                var u2 = random.NextSingle();
                var radius = MathF.Sqrt(-2.0f * MathF.Log(u1));
                var theta = 2.0f * MathF.PI * u2;
                var standardNormalRand = radius * MathF.Cos(theta);
                latents.SetValue(i, standardNormalRand * initNoiseSigma);
            }
            return latents;
        }


        /// <summary>
        /// Gets the total product for the specified dimensions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <param name="startIndex">The start index.</param>
        /// <returns>System.Int64.</returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static long GetProduct(this ReadOnlySpan<int> dimensions, int startIndex = 0)
        {
            long product = 1;
            for (int i = startIndex; i < dimensions.Length; i++)
            {
                if (dimensions[i] < 0)
                    throw new ArgumentOutOfRangeException($"{nameof(dimensions)}[{i}]");

                product *= dimensions[i];
            }
            return product;
        }


        /// <summary>
        /// Gets the strides for the specified dimensions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>System.Int32[].</returns>
        public static int[] GetStrides(this ReadOnlySpan<int> dimensions)
        {
            var strides = new int[dimensions.Length];
            if (dimensions.Length == 0)
                return strides;

            int stride = 1;
            for (int i = strides.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= dimensions[i];
            }
            return strides;
        }


        /// <summary>
        /// Gets the tensor index with the specified indices and strides.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <param name="strides">The strides.</param>
        /// <param name="startFromDimension">The start from dimension.</param>
        /// <returns>System.Int32.</returns>
        public static int GetIndex(this ReadOnlySpan<int> indices, ReadOnlySpan<int> strides, int startFromDimension = 0)
        {
            int index = 0;
            for (int i = startFromDimension; i < indices.Length; i++)
            {
                index += strides[i] * indices[i];
            }
            return index;
        }


        /// <summary>
        /// Normalizes the values from range -1 to 1 to 0 to 1.
        /// </summary>
        /// <param name="span">The span.</param>
        public static void NormalizeOneOneToZeroOne(this Span<float> span)
        {
            for (int i = 0; i < span.Length; i++)
            {
                span[i] = Math.Clamp(span[i] / 2f + 0.5f, 0f, 1f);
            }
        }


        /// <summary>
        /// Normalizes the values from range 0 to 1 to -1 to 1.
        /// </summary>
        /// <param name="span">The span.</param>
        public static void NormalizeZeroOneToOneOne(this Span<float> span)
        {
            for (int i = 0; i < span.Length; i++)
            {
                span[i] = Math.Clamp(2f * span[i] - 1f, -1f, 1f);
            }
        }


        /// <summary>
        /// Min/Max normalizaton to zero to one.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>Span&lt;System.Single&gt;.</returns>
        public static Span<float> NormalizeMinMaxToZeroToOne(this Span<float> values)
        {
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            for (int i = 0; i < values.Length; i++)
            {
                float value = values[i];
                if (value < min) min = value;
                if (value > max) max = value;
            }

            float range = max - min;
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Math.Clamp((values[i] - min) / range, 0f, 1f);
            }
            return values;
        }


        /// <summary>
        /// Min/Max normalizaton to one to one.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>Span&lt;System.Single&gt;.</returns>
        public static Span<float> NormalizeMinMaxToOneToOne(this Span<float> values)
        {
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            for (int i = 0; i < values.Length; i++)
            {
                float value = values[i];
                if (value < min)
                    min = value;
                if (value > max)
                    max = value;
            }

            float range = max - min;
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Math.Clamp(2 * (values[i] - min) / range - 1, -1f, 1f);
            }
            return values;
        }


        /// <summary>
        /// Normalizes the tensor values from range 1 to 1 to 0 to 1.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public static void NormalizeOneOneToZeroOne(this Tensor<float> tensor)
        {
            tensor.Memory.Span.NormalizeOneOneToZeroOne();
        }


        /// <summary>
        /// Normalizes the tensor values from range 0 to 1 to -1 to 1.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public static void NormalizeZeroOneToOneOne(this Tensor<float> tensor)
        {
            tensor.Memory.Span.NormalizeZeroOneToOneOne();
        }

        /// <summary>
        /// Tensor filled with ones
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public static Tensor<T> Ones<T>(ReadOnlySpan<int> dimensions) where T : INumber<T> => Fill(dimensions, T.One);


        /// <summary>
        /// Tensor filled with zeros
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public static Tensor<T> Zeros<T>(ReadOnlySpan<int> dimensions) where T : INumber<T> => Fill(dimensions, T.Zero);


        /// <summary>
        /// Tensor filled with specified value
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        public static Tensor<T> Fill<T>(ReadOnlySpan<int> dimensions, T value) where T : INumber<T>
        {
            var result = new Tensor<T>(dimensions);
            result.Fill(value);
            return result;
        }


        /// <summary>
        /// Concatenates the specified tensors along the specified axis.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <param name="axis">The axis.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException">Only axis 0,1,2 is supported</exception>
        public static Tensor<T> Concatenate<T>(this Tensor<T> tensor1, Tensor<T> tensor2, int axis = 0)
        {
            if (tensor1 == null)
                return tensor2.Clone();

            return axis switch
            {
                0 => ConcatenateAxis0(tensor1, tensor2),
                1 => ConcatenateAxis1(tensor1, tensor2),
                2 => ConcatenateAxis2(tensor1, tensor2),
                _ => throw new NotImplementedException("Only axis 0, 1, 2 is supported")
            };
        }


        /// <summary>
        /// Concatenates Axis 0.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        private static Tensor<T> ConcatenateAxis0<T>(this Tensor<T> tensor1, Tensor<T> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[0] += tensor2.Dimensions[0];

            var buffer = new Tensor<T>(dimensions);
            tensor1.Memory.Span.CopyTo(buffer.Memory.Span[..(int)tensor1.Length]);
            tensor2.Memory.Span.CopyTo(buffer.Memory.Span[(int)tensor1.Length..]);
            return buffer;
        }


        /// <summary>
        /// Concatenates Axis 1.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        /// <exception cref="System.ArgumentException">Length 2, 3 or 4 currently supported</exception>
        private static Tensor<T> ConcatenateAxis1<T>(Tensor<T> tensor1, Tensor<T> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[1] += tensor2.Dimensions[1];
            var concatenatedTensor = new Tensor<T>(dimensions);

            if (tensor1.Dimensions.Length == 2)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        concatenatedTensor[i, j] = tensor1[i, j];

                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        concatenatedTensor[i, j + tensor1.Dimensions[1]] = tensor2[i, j];
            }
            else if (tensor1.Dimensions.Length == 3)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        for (int k = 0; k < tensor1.Dimensions[2]; k++)
                            concatenatedTensor[i, j, k] = tensor1[i, j, k];

                for (int i = 0; i < tensor2.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        for (int k = 0; k < tensor2.Dimensions[2]; k++)
                            concatenatedTensor[i, j + tensor1.Dimensions[1], k] = tensor2[i, j, k];
            }
            else if (tensor1.Dimensions.Length == 4)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        for (int k = 0; k < tensor1.Dimensions[2]; k++)
                            for (int l = 0; l < tensor1.Dimensions[3]; l++)
                                concatenatedTensor[i, j, k, l] = tensor1[i, j, k, l];

                for (int i = 0; i < tensor2.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        for (int k = 0; k < tensor2.Dimensions[2]; k++)
                            for (int l = 0; l < tensor2.Dimensions[3]; l++)
                                concatenatedTensor[i, j + tensor1.Dimensions[1], k, l] = tensor2[i, j, k, l];
            }
            else
            {
                throw new ArgumentException("Length 2, 3 or 4 currently supported");
            }
            return concatenatedTensor;
        }


        /// <summary>
        /// Concatenates Axis 2.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <returns>Tensor&lt;T&gt;.</returns>
        private static Tensor<T> ConcatenateAxis2<T>(Tensor<T> tensor1, Tensor<T> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[2] += tensor2.Dimensions[2];
            var concatenatedTensor = new Tensor<T>(dimensions);

            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < dimensions[1]; j++)
                    for (int k = 0; k < tensor1.Dimensions[2]; k++)
                        concatenatedTensor[i, j, k] = tensor1[i, j, k];

            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < dimensions[1]; j++)
                    for (int k = 0; k < tensor2.Dimensions[2]; k++)
                        concatenatedTensor[i, j, k + tensor1.Dimensions[2]] = tensor2[i, j, k];

            return concatenatedTensor;
        }


        /// <summary>
        /// Computes the softmax function over the specified tensor
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> SoftMax(this Tensor<float> tensor)
        {
            TensorPrimitives.SoftMax(tensor.Memory.Span, tensor.Memory.Span);
            return tensor;
        }


        /// <summary>
        /// Resizes the specified ImageTensor (Bilinear)
        /// </summary>
        /// <param name="sourceImage">The input.</param>
        /// <param name="targetWidth">Width of the target.</param>
        /// <param name="targetHeight">Height of the target.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageTensor ResizeImage(this ImageTensor sourceImage, int targetWidth, int targetHeight, ResizeMode resizeMode)
        {
            var cropX = 0;
            var cropY = 0;
            var croppedWidth = targetWidth;
            var croppedHeight = targetWidth;
            var channels = sourceImage.Dimensions[1];
            var sourceHeight = sourceImage.Dimensions[2];
            var sourceWidth = sourceImage.Dimensions[3];
            var destination = new ImageTensor(new[] { 1, channels, targetHeight, targetWidth });
            if (resizeMode == ResizeMode.Crop)
            {
                var scaleX = (float)targetWidth / sourceImage.Width;
                var scaleY = (float)targetHeight / sourceImage.Height;
                var scaleFactor = Math.Max(scaleX, scaleY);
                croppedWidth = (int)(sourceImage.Width * scaleFactor);
                croppedHeight = (int)(sourceImage.Height * scaleFactor);
                cropX = Math.Abs(Math.Max((croppedWidth - targetWidth) / 2, 0));
                cropY = Math.Abs(Math.Max((croppedHeight - targetHeight) / 2, 0));
            }

            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < croppedHeight; h++)
                {
                    for (int w = 0; w < croppedWidth; w++)
                    {
                        // Map target pixel to input pixel
                        var y = h * (float)(sourceHeight - 1) / (croppedHeight - 1);
                        var x = w * (float)(sourceWidth - 1) / (croppedWidth - 1);

                        var y0 = (int)Math.Floor(y);
                        var x0 = (int)Math.Floor(x);
                        var y1 = Math.Min(y0 + 1, sourceHeight - 1);
                        var x1 = Math.Min(x0 + 1, sourceWidth - 1);

                        // Bilinear interpolation
                        var dy = y - y0;
                        var dx = x - x0;
                        var topLeft = sourceImage[0, c, y0, x0];
                        var topRight = sourceImage[0, c, y0, x1];
                        var bottomLeft = sourceImage[0, c, y1, x0];
                        var bottomRight = sourceImage[0, c, y1, x1];

                        var targetY = h - cropY;
                        var targetX = w - cropX;
                        if (targetX >= 0 && targetY >= 0 && targetY < destination.Height && targetX < destination.Width)
                        {
                            destination[0, c, targetY, targetX] =
                                    topLeft * (1 - dx) * (1 - dy) +
                                    topRight * dx * (1 - dy) +
                                    bottomLeft * (1 - dx) * dy +
                                    bottomRight * dx * dy;
                        }
                    }
                }
            }
            return destination;
        }

    }
}
