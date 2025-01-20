// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Extractors
{
    public static class Extensions
    {
        /// <summary>
        /// Inverts the values.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>Span&lt;System.Single&gt;.</returns>
        public static Span<float> Invert(this Span<float> values)
        {
            for (int j = 0; j < values.Length; j++)
            {
                values[j] = -values[j];
            }
            return values;
        }
    }
}
