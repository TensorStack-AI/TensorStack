// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;
using TensorStack.Common.Vision;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace TensorStack.Common
{
    /// <summary>
    /// Helper extensions for Tensor and TensorSpan, Math, Copy etc.
    /// </summary>
    public static class TensorExtensions
    {
        #region Divide

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
        /// Divides the specified value
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Divide(this Tensor<float> tensor, float value)
        {
            TensorPrimitives.Divide(tensor.Span, value, tensor.Memory.Span);
            return tensor;
        }

        /// <summary>
        /// COPY: Divides the specified value to new tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> DivideTo(this Tensor<float> tensor, float value)
        {
            var result = new Tensor<float>(tensor.Dimensions);
            TensorPrimitives.Divide(tensor.Span, value, result.Memory.Span);
            return result;
        }

        #endregion

        #region Multiply

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
        /// Multiplies each Tensor value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Multiply(this Tensor<float> tensor, float value)
        {
            TensorPrimitives.Multiply(tensor.Span, value, tensor.Memory.Span);
            return tensor;
        }


        /// <summary>
        /// COPY: Multiplies each Tensor value to new tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> MultiplyTo(this Tensor<float> tensor, float value)
        {
            var result = new Tensor<float>(tensor.Dimensions);
            TensorPrimitives.Multiply(tensor.Span, value, result.Memory.Span);
            return result;
        }

        #endregion

        #region Add

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
        /// Adds TensorB to tensorA
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Add(this Tensor<float> tensorA, Tensor<float> tensorB)
        {
            TensorPrimitives.Add(tensorA.Span, tensorB.Span, tensorA.Memory.Span);
            return tensorA;
        }


        /// <summary>
        /// Adds TensorB to tensorA to new tensor
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> AddTo(this Tensor<float> tensorA, Tensor<float> tensorB)
        {
            var result = new Tensor<float>(tensorA.Dimensions);
            TensorPrimitives.Add(tensorA.Span, tensorB.Span, result.Memory.Span);
            return result;
        }


        /// <summary>
        /// Adds the specified value to each Tensor value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Add(this Tensor<float> tensor, float value)
        {
            TensorPrimitives.Add(tensor.Span, value, tensor.Memory.Span);
            return tensor;
        }


        /// <summary>
        /// Adds the value to the Tensor to new tensor
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> AddTo(this Tensor<float> tensor, float value)
        {
            var result = new Tensor<float>(tensor.Dimensions);
            TensorPrimitives.Add(tensor.Span, value, result.Memory.Span);
            return result;
        }

        #endregion

        #region Subtract

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
        /// COPY: Subtracts TensorB from TensorA to a new tensor
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> SubtractTo(this TensorSpan<float> tensorA, TensorSpan<float> tensorB)
        {
            var result = new TensorSpan<float>(tensorA.Dimensions);
            TensorPrimitives.Subtract(tensorA.Span, tensorB.Span, result.Span);
            return result;
        }


        /// <summary>
        /// Subtracts the value from the Tensor
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
        /// COPY: Subtracts the value from the Tensor to a new tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public static TensorSpan<float> SubtractTo(this TensorSpan<float> tensor, float value)
        {
            var result = new TensorSpan<float>(tensor.Dimensions);
            TensorPrimitives.Subtract(tensor.Span, value, result.Span);
            return result;
        }


        /// <summary>
        /// Subtracts TensorB from TensorA
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Subtract(this Tensor<float> tensorA, Tensor<float> tensorB)
        {
            TensorPrimitives.Subtract(tensorA.Span, tensorB.Span, tensorA.Memory.Span);
            return tensorA;
        }


        /// <summary>
        /// COPY: Subtracts TensorB from TensorA to a new tensor
        /// </summary>
        /// <param name="tensorA">The tensor a.</param>
        /// <param name="tensorB">The tensor b.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> SubtractTo(this Tensor<float> tensorA, Tensor<float> tensorB)
        {
            var result = new Tensor<float>(tensorA.Dimensions);
            TensorPrimitives.Subtract(tensorA.Span, tensorB.Span, result.Memory.Span);
            return result;
        }


        /// <summary>
        /// Subtracts the specified value from the Tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Subtract(this Tensor<float> tensor, float value)
        {
            TensorPrimitives.Subtract(tensor.Span, value, tensor.Memory.Span);
            return tensor;
        }


        /// <summary>
        /// COPY: Subtracts the specified value from the Tensor to a new tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> SubtractTo(this Tensor<float> tensor, float value)
        {
            var result = new Tensor<float>(tensor.Dimensions);
            TensorPrimitives.Subtract(tensor.Span, value, result.Memory.Span);
            return result;
        }

        #endregion


        /// <summary>
        /// Sums the tensors.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <param name="dimensions">The dimensions.</param>
        public static Tensor<float> SumTensors(this Tensor<float>[] tensors, ReadOnlySpan<int> dimensions)
        {
            var result = new Tensor<float>(dimensions);
            for (int m = 0; m < tensors.Length; m++)
            {
                TensorPrimitives.Add(result.Span, tensors[m].Span, result.Memory.Span);
            }
            return result;
        }


        /// <summary>
        /// Clips to the specified minimum/maximum value.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="minValue">The minimum value.</param>
        /// <param name="maxValue">The maximum value.</param>
        public static Tensor<float> ClipTo(this Tensor<float> tensor, float minValue, float maxValue)
        {
            var clipTensor = new Tensor<float>(tensor.Dimensions);
            for (int i = 0; i < tensor.Length; i++)
            {
                clipTensor.SetValue(i, Math.Clamp(tensor.Memory.Span[i], minValue, maxValue));
            }
            return clipTensor;
        }


        /// <summary>
        /// Split first tensor from batch and return
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns></returns>
        public static Tensor<T> FirstBatch<T>(this Tensor<T> tensor)
        {
            return Split(tensor).FirstOrDefault();
        }


        /// <summary>
        /// Reshapes to new tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> ReshapeTo(this Tensor<float> tensor, ReadOnlySpan<int> dimensions)
        {
            return new Tensor<float>(tensor.Memory.ToArray(), dimensions);
        }


        /// <summary>
        /// Reshapes to the specified dimensions.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        public static Tensor<float> Reshape(this Tensor<float> tensor, ReadOnlySpan<int> dimensions)
        {
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
        public static IEnumerable<Tensor<T>> Split<T>(this Tensor<T> tensor, int axis = 0)
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
                yield return new Tensor<T>(tensor.Memory.Slice(start, newLength), dimensions);
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
        public static Tensor<float> NextTensor(this Random random, ReadOnlySpan<int> dimensions, float initialvalue = 1f)
        {
            var latents = new Tensor<float>(dimensions);
            for (int i = 0; i < latents.Length; i++)
            {
                var u1 = random.NextSingle();
                var u2 = random.NextSingle();
                var radius = MathF.Sqrt(-2.0f * MathF.Log(u1));
                var theta = 2.0f * MathF.PI * u2;
                var standardNormalRand = radius * MathF.Cos(theta);
                latents.SetValue(i, standardNormalRand * initialvalue);
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
        public static void NormalizeZeroOne(this Span<float> span)
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
        public static void NormalizeOneOne(this Span<float> span)
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
        public static Tensor<float> NormalizeZeroOne(this Tensor<float> tensor)
        {
            tensor.Memory.Span.NormalizeZeroOne();
            return tensor;
        }


        /// <summary>
        /// Normalizes the tensor values from range 0 to 1 to -1 to 1.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public static Tensor<float> NormalizeOneOne(this Tensor<float> tensor)
        {
            tensor.Memory.Span.NormalizeOneOne();
            return tensor;
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
        /// Lerps the specified valuse.
        /// </summary>
        /// <param name="span1">The span1.</param>
        /// <param name="span2">The span2.</param>
        /// <param name="value">The value.</param>
        public static void Lerp(this Memory<float> span1, Memory<float> span2, float value)
        {
            TensorPrimitives.Lerp(span1.Span, span2.Span, value, span1.Span);
        }


        /// <summary>
        /// Inverts the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public static Tensor<float> Invert(this Tensor<float> tensor)
        {
            var result = new Tensor<float>(tensor.Dimensions);
            for (int j = 0; j < result.Length; j++)
                result.SetValue(j, 1f - tensor.GetValue(j));
            return result;
        }


        /// <summary>
        /// Return tensor with guidance dimension if required
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="applyGuidance">if set to <c>true</c> [apply guidance].</param>
        public static Tensor<float> WithGuidance(this Tensor<float> tensor, bool applyGuidance)
        {
            if (!applyGuidance)
                return tensor;

            return tensor.Repeat(2);
        }


        /// <summary>
        /// Resizes the specified ImageTensor
        /// </summary>
        /// <param name="sourceImage">The source image.</param>
        /// <param name="targetWidth">Width of the target.</param>
        /// <param name="targetHeight">Height of the target.</param>
        /// <param name="resizeMode">The resize mode.</param>
        /// <param name="resizeMethod">The resize method.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageTensor ResizeImage(this ImageTensor sourceImage, int targetWidth, int targetHeight, ResizeMode resizeMode = ResizeMode.Stretch, ResizeMethod resizeMethod = ResizeMethod.Bicubic)
        {
            return resizeMethod switch
            {
                ResizeMethod.Bicubic => ResizeImageBicubic(sourceImage, targetWidth, targetHeight, resizeMode),
                _ => ResizeImageBilinear(sourceImage, targetWidth, targetHeight, resizeMode),
            };
        }


        /// <summary>
        /// Resizes the specified ImageTensor (Bilinear)
        /// </summary>
        /// <param name="sourceImage">The input.</param>
        /// <param name="targetWidth">Width of the target.</param>
        /// <param name="targetHeight">Height of the target.</param>
        /// <returns>ImageTensor.</returns>
        private static ImageTensor ResizeImageBilinear(ImageTensor sourceImage, int targetWidth, int targetHeight, ResizeMode resizeMode)
        {
            var channels = sourceImage.Dimensions[1];
            var sourceHeight = sourceImage.Dimensions[2];
            var sourceWidth = sourceImage.Dimensions[3];
            var cropSize = GetCropCoordinates(sourceHeight, sourceWidth, targetHeight, targetWidth, resizeMode);
            var destination = new ImageTensor(new[] { 1, channels, targetHeight, targetWidth });
            Parallel.For(0, channels, c =>
            {
                for (int h = 0; h < cropSize.MaxY; h++)
                {
                    for (int w = 0; w < cropSize.MaxX; w++)
                    {
                        var y = h * (float)(sourceHeight - 1) / (cropSize.MaxY - 1);
                        var x = w * (float)(sourceWidth - 1) / (cropSize.MaxX - 1);

                        var y0 = (int)Math.Floor(y);
                        var x0 = (int)Math.Floor(x);
                        var y1 = Math.Min(y0 + 1, sourceHeight - 1);
                        var x1 = Math.Min(x0 + 1, sourceWidth - 1);

                        var dy = y - y0;
                        var dx = x - x0;
                        var topLeft = sourceImage[0, c, y0, x0];
                        var topRight = sourceImage[0, c, y0, x1];
                        var bottomLeft = sourceImage[0, c, y1, x0];
                        var bottomRight = sourceImage[0, c, y1, x1];

                        var targetY = h - cropSize.MinY;
                        var targetX = w - cropSize.MinX;
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
            });
            return destination;
        }


        /// <summary>
        /// Resizes the specified ImageTensor (ResizeImageBicubic)
        /// </summary>
        /// <param name="sourceImage">The input.</param>
        /// <param name="targetWidth">Width of the target.</param>
        /// <param name="targetHeight">Height of the target.</param>
        /// <returns>ImageTensor.</returns>
        private static ImageTensor ResizeImageBicubic(ImageTensor sourceImage, int targetWidth, int targetHeight, ResizeMode resizeMode = ResizeMode.Stretch)
        {
            var channels = sourceImage.Dimensions[1];
            var sourceHeight = sourceImage.Dimensions[2];
            var sourceWidth = sourceImage.Dimensions[3];
            var cropSize = GetCropCoordinates(sourceHeight, sourceWidth, targetHeight, targetWidth, resizeMode);
            var destination = new ImageTensor(new[] { 1, channels, targetHeight, targetWidth });
            Parallel.For(0, channels, c =>
            {
                for (int h = 0; h < cropSize.MaxY; h++)
                {
                    for (int w = 0; w < cropSize.MaxX; w++)
                    {
                        float y = h * (float)(sourceHeight - 1) / (cropSize.MaxY - 1);
                        float x = w * (float)(sourceWidth - 1) / (cropSize.MaxX - 1);

                        int yInt = (int)Math.Floor(y);
                        int xInt = (int)Math.Floor(x);
                        float yFrac = y - yInt;
                        float xFrac = x - xInt;

                        float[] colVals = new float[4];

                        for (int i = -1; i <= 2; i++)
                        {
                            int yi = Math.Clamp(yInt + i, 0, sourceHeight - 1);
                            float[] rowVals = new float[4];

                            for (int j = -1; j <= 2; j++)
                            {
                                int xi = Math.Clamp(xInt + j, 0, sourceWidth - 1);
                                rowVals[j + 1] = sourceImage[0, c, yi, xi];
                            }

                            colVals[i + 1] = CubicInterpolate(rowVals[0], rowVals[1], rowVals[2], rowVals[3], xFrac);
                        }

                        var targetY = h - cropSize.MinY;
                        var targetX = w - cropSize.MinX;
                        if (targetX >= 0 && targetY >= 0 && targetY < targetHeight && targetX < targetWidth)
                        {
                            destination[0, c, h, w] = CubicInterpolate(colVals[0], colVals[1], colVals[2], colVals[3], yFrac);
                        }
                    }
                }
            });

            return destination;
        }


        /// <summary>
        /// Cubic interpolate.
        /// </summary>
        /// <param name="v0">The v0.</param>
        /// <param name="v1">The v1.</param>
        /// <param name="v2">The v2.</param>
        /// <param name="v3">The v3.</param>
        /// <param name="fraction">The fraction.</param>
        /// <returns>System.Single.</returns>
        private static float CubicInterpolate(float v0, float v1, float v2, float v3, float fraction)
        {
            float A = (-0.5f * v0) + (1.5f * v1) - (1.5f * v2) + (0.5f * v3);
            float B = (v0 * -1.0f) + (v1 * 2.5f) - (v2 * 2.0f) + (v3 * 0.5f);
            float C = (-0.5f * v0) + (0.5f * v2);
            float D = v1;
            return A * (fraction * fraction * fraction) + B * (fraction * fraction) + C * fraction + D;
        }


        /// <summary>
        /// Gets the crop coordinates.
        /// </summary>
        /// <param name="sourceHeight">Height of the source.</param>
        /// <param name="sourceWidth">Width of the source.</param>
        /// <param name="targetHeight">Height of the target.</param>
        /// <param name="targetWidth">Width of the target.</param>
        /// <param name="resizeMode">The resize mode.</param>
        /// <returns>CoordinateBox&lt;System.Int32&gt;.</returns>
        private static CoordinateBox<int> GetCropCoordinates(int sourceHeight, int sourceWidth, int targetHeight, int targetWidth, ResizeMode resizeMode)
        {
            var cropX = 0;
            var cropY = 0;
            var croppedWidth = targetWidth;
            var croppedHeight = targetHeight;
            if (resizeMode == ResizeMode.Crop)
            {
                var scaleX = (float)targetWidth / sourceWidth;
                var scaleY = (float)targetHeight / sourceHeight;
                var scaleFactor = Math.Max(scaleX, scaleY);
                croppedWidth = (int)(sourceWidth * scaleFactor);
                croppedHeight = (int)(sourceHeight * scaleFactor);
                cropX = Math.Abs(Math.Max((croppedWidth - targetWidth) / 2, 0));
                cropY = Math.Abs(Math.Max((croppedHeight - targetHeight) / 2, 0));
            }
            return new CoordinateBox<int>(cropX, cropY, croppedWidth, croppedHeight);
        }
    }
}
