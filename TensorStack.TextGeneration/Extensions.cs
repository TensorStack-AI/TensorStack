// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;

namespace TensorStack.TextGeneration
{
    public static class Extensions
    {
        /// <summary>
        /// Disposes the specified collection of disposables.
        /// </summary>
        /// <param name="disposable">The disposable.</param>
        public static void Dispose(this IEnumerable<IDisposable> disposable)
        {
            foreach (var item in disposable)
            {
                item?.Dispose();
            }
        }
    }
}