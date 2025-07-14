// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Linq;

namespace TensorStack.Common
{
    public sealed record ModelMetadata
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ModelMetadata"/> class.
        /// </summary>
        /// <param name="session">The session.</param>
        public ModelMetadata(InferenceSession session)
        {
            Inputs = session.InputMetadata
                .Select(NamedMetadata.Create)
                .ToList();
            Outputs = session.OutputMetadata
                .Select(NamedMetadata.Create)
                .ToList();
        }

        /// <summary>
        /// Gets or sets the inputs.
        /// </summary>
        public IReadOnlyList<NamedMetadata> Inputs { get; }

        /// <summary>
        /// Gets or sets the outputs.
        /// </summary>
        public IReadOnlyList<NamedMetadata> Outputs { get; }
    }
}
