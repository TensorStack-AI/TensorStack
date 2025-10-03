// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;

namespace TensorStack.Common
{
    public sealed record NamedMetadata(string Name, NodeMetadata Value)
    {
        public ReadOnlySpan<int> Dimensions => Value.Dimensions;
        public Type DataType => Value.ElementType;
        public TensorElementType ElementType => Value.ElementDataType;

        /// <summary>
        /// Creates the specified metadata.
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <returns>NamedMetadata.</returns>
        internal static NamedMetadata Create(KeyValuePair<string, NodeMetadata> metadata)
        {
            return new NamedMetadata(metadata.Key, metadata.Value);
        }
    }
}
