// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace TensorStack.Common
{
    public static class Extensions
    {
        /// <summary>
        /// Converts to long.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        public static long[] ToLong(this ReadOnlySpan<int> array)
        {
            return Array.ConvertAll(array.ToArray(), Convert.ToInt64);
        }


        /// <summary>
        /// Converts the string representation of a number to an integer.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        public static int[] ToInt(this long[] array)
        {
            return Array.ConvertAll(array, Convert.ToInt32);
        }


        /// <summary>
        /// Converts to intsafe.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        /// <exception cref="OverflowException">$"Value {value} at index {i} is outside the range of an int.</exception>
        public static int[] ToIntSafe(this long[] array)
        {
            var result = GC.AllocateUninitializedArray<int>(array.Length);
            for (int i = 0; i < array.Length; i++)
            {
                long value = array[i];

                if (value < int.MinValue || value > int.MaxValue)
                    value = 0;

                result[i] = (int)value;
            }

            return result;
        }


        /// <summary>
        /// Converts to long.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns></returns>
        public static long[] ToLong(this int[] array)
        {
            return Array.ConvertAll(array, Convert.ToInt64);
        }


        /// <summary>
        /// Determines whether the the source sequence is null or empty
        /// </summary>
        /// <typeparam name="TSource">Type of elements in <paramref name="source" /> sequence.</typeparam>
        /// <param name="source">The source sequence.</param>
        /// <returns>
        ///   <c>true</c> if the source sequence is null or empty; otherwise, <c>false</c>.
        /// </returns>
        public static bool IsNullOrEmpty<TSource>(this IEnumerable<TSource> source)
        {
            return source == null || !source.Any();
        }


        /// <summary>
        /// Get the item at the specifed index.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list">The list.</param>
        /// <param name="item">The item.</param>
        /// <returns>System.Int32.</returns>
        public static int IndexOf<T>(this IReadOnlyList<T> list, T item) where T : IEquatable<T>
        {
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i].Equals(item))
                    return i;
            }
            return -1;
        }


        public static int IndexOf<T>(this IReadOnlyList<T> list, Func<T, bool> itemSelector) where T : IEquatable<T>
        {
            var item = list.FirstOrDefault(itemSelector);
            if (item == null)
                return -1;

            return IndexOf(list, item);
        }


        public static T[] PadOrTruncate<T>(this T[] inputs, T padValue, int requiredLength)
        {
            var result = new T[requiredLength];
            var countToCopy = Math.Min(inputs.Length, requiredLength);
            Array.Copy(inputs, result, countToCopy);
            if (inputs.Length < requiredLength)
            {
                for (int i = inputs.Length; i < requiredLength; i++)
                    result[i] = padValue;
            }
            return result;
        }


        public static T[] Pad<T>(this T[] inputs, T padValue, int requiredLength)
        {
            int count = inputs.Length;
            if (requiredLength <= count)
                return inputs;

            var result = new T[requiredLength];
            Array.Copy(inputs, result, count);
            for (int i = count; i < requiredLength; i++)
                result[i] = padValue;

            return result;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ZeroIfNan(this float value)
        {
            return float.IsNaN(value) ? 0f : value;
        }
    }
}
