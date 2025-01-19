// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace TensorStack.Core.Inference
{
    public sealed record NamedMetadata(string Name, NodeMetadata Value)
    {
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
