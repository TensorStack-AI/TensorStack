// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using TensorStack.Common.Tensor;

namespace TensorStack.Transformers
{
    public static class Extensions
    {
        public static Span<T> GetBatchAsSpan<T>(this Tensor<T> tensor, int batch)
        {
            var size = tensor.Dimensions.Length == 2
                ? tensor.Dimensions[1]
                : tensor.Dimensions[2];
            return tensor.Memory.Span.Slice(batch * size, size);
        }


        public static List<T>[] Repeat<T>(this List<T> value, int count)
        {
            if (count == 1)
                return [value];

            var result = new List<T>[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = new List<T>(value);
            }
            return result;
        }
    }
}